import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from piq import LPIPS

from cm.data.loader import create_data_loader
from cm.utils.checkpoint import save_checkpoint, load_checkpoint
from cm.training.ema import update_ema
from cm.training.losses import consistency_training_loss
from cm.diffusion.karras_schedule import n_schedule, mu_schedule


class CTTrainer:
    def __init__(
        self,
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        data_dir: str,
        resume_ckpt: str = None,
        batch_size: int = 512,
        image_size: int = 32,
        lr: float = 2e-4,
        max_steps: int = 800_000,
        save_interval: int = 5000,
        # CT-specific schedule knobs (Song et al. 2023, Section 5)
        s0: int = 2,
        s1: int = 150,
        mu0: float = 0.95,
        use_lpips: bool = True,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_steps = max_steps
        self.save_interval = save_interval
        self.use_lpips = use_lpips
        self.s0 = s0
        self.s1 = s1
        self.mu0 = mu0

        self._setup_ddp()

        # 1. Models — target starts frozen, will be EMA-updated each step
        self.online_model = online_model
        self.target_model = target_model
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        if self.is_distributed:
            self.online_model = DDP(
                self.online_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # 2. Optimizer + AMP scaler
        online_params = (
            self.online_model.module.parameters()
            if self.is_distributed
            else self.online_model.parameters()
        )
        self.optimizer = AdamW(online_params, lr=lr)
        self.scaler = GradScaler(self.device.type)

        self.start_step = 0
        if resume_ckpt and os.path.exists(resume_ckpt):
            self._resume_training(resume_ckpt)

        # 3. Infinite data generator
        self.data_generator = create_data_loader(
            data_dir=data_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
        )

        # 4. LPIPS distance metric (paper's default d(·,·) on CIFAR-10)
        if self.use_lpips:
            self.lpips_fn = LPIPS(replace_pooling=True, reduction="none").to(self.device)
        else:
            self.lpips_fn = None

    def _setup_ddp(self):
        self.is_distributed = "WORLD_SIZE" in os.environ
        if self.is_distributed:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.is_main_process = (self.local_rank == 0)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.is_main_process = True

    def _resume_training(self, resume_ckpt: str):
        if self.is_main_process:
            print(f"[*] Resuming training from {resume_ckpt}...")
        self.start_step, _ = load_checkpoint(
            load_path=resume_ckpt,
            ema_model=self.target_model,
            model=self.online_model.module if self.is_distributed else self.online_model,
            optimizer=self.optimizer,
            device=str(self.device),
        )

    def train(self):
        if self.is_main_process:
            print("[*] Starting Consistency Training (no teacher)...")

        for step in range(self.start_step, self.max_steps):
            self.online_model.train()
            self.optimizer.zero_grad()

            # 1. Step-dependent schedules — the heart of CT
            N_k  = n_schedule(step, self.max_steps, self.s0, self.s1)
            mu_k = mu_schedule(step, self.max_steps, self.mu0, self.s0, self.s1)

            # 2. Fetch a batch
            images, _ = next(self.data_generator)
            images = images.to(self.device)

            # 3. CT loss under AMP
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                loss = consistency_training_loss(
                    online_model=self.online_model,
                    target_model=self.target_model,
                    images=images,
                    num_scales=N_k,
                    use_lpips=self.use_lpips,
                    lpips_loss_fn=self.lpips_fn,
                )

            # 4. Backward + optimizer step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 5. EMA target update with this step's μ(k)
            update_ema(self.target_model, self.online_model, mu=mu_k)

            # 6. Logging & checkpointing
            if self.is_main_process and step % 100 == 0:
                print(
                    f"Step {step}/{self.max_steps} | "
                    f"N(k)={N_k} | μ(k)={mu_k:.5f} | "
                    f"CT Loss: {loss.item():.4f}"
                )

            if self.is_main_process and step > 0 and step % self.save_interval == 0:
                save_checkpoint(
                    save_path=f"checkpoints/step_{step:06d}.pt",
                    step=step,
                    ema_model=self.target_model,
                    model=self.online_model.module if self.is_distributed else self.online_model,
                    optimizer=self.optimizer,
                )

        if self.is_distributed:
            dist.destroy_process_group()
