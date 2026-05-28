import os

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import RAdam
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
        sampling_ema_model: torch.nn.Module,
        data_dir: str,
        resume_ckpt: str | None = None,
        batch_size: int = 512,
        image_size: int = 32,
        lr: float = 2e-4,
        max_steps: int = 800_000,
        save_interval: int = 5000,
        s0: int = 2,
        s1: int = 150,
        mu0: float = 0.95,
        use_lpips: bool = True,
        log_every: int = 50,
        use_fp16: bool = False,
        sampling_ema_decay: float = 0.9999,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_steps = max_steps
        self.save_interval = save_interval
        self.use_lpips = use_lpips
        self.s0 = s0
        self.s1 = s1
        self.mu0 = mu0
        self.log_every = log_every
        self.use_fp16 = use_fp16
        self.sampling_ema_decay = sampling_ema_decay

        self._setup_ddp()

        self.online_model = online_model
        self.target_model = target_model
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        self.sampling_ema_model = sampling_ema_model
        self.sampling_ema_model.requires_grad_(False)
        self.sampling_ema_model.eval()

        if self.is_distributed:
            self.online_model = DDP(
                self.online_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        online_params = (
            self.online_model.module.parameters()
            if self.is_distributed
            else self.online_model.parameters()
        )
        self.optimizer = RAdam(online_params, lr=lr, weight_decay=0.0)
        self.scaler = GradScaler(self.device.type, enabled=self.use_fp16)

        self.start_step = 0
        if resume_ckpt and os.path.exists(resume_ckpt):
            self._resume_training(resume_ckpt)

        self.data_generator = create_data_loader(
            data_dir=data_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
        )

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
            sampling_ema_model=self.sampling_ema_model,
            device=str(self.device),
        )

    def train(self):
        if self.is_main_process:
            print("[*] Starting Consistency Training (no teacher)...")

        for step in range(self.start_step, self.max_steps):
            self.online_model.train()
            self.optimizer.zero_grad()

            N_k  = n_schedule(step, self.max_steps, self.s0, self.s1)
            mu_k = mu_schedule(step, self.max_steps, self.mu0, self.s0, self.s1)

            images, _ = next(self.data_generator)
            images = images.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_fp16):
                loss = consistency_training_loss(
                    online_model=self.online_model,
                    target_model=self.target_model,
                    images=images,
                    num_scales=N_k,
                    use_lpips=self.use_lpips,
                    lpips_loss_fn=self.lpips_fn,
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            update_ema(self.target_model, self.online_model, mu=mu_k)
            update_ema(self.sampling_ema_model, self.online_model, mu=self.sampling_ema_decay)

            if self.is_main_process and step % self.log_every == 0:
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
                    sampling_ema_model=self.sampling_ema_model,
                )

        if self.is_distributed:
            dist.destroy_process_group()
