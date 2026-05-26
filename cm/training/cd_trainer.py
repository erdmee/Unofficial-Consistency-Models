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
from cm.training.losses import consistency_distillation_loss

class CDTrainer:
    def __init__(
        self,
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        teacher_model: torch.nn.Module,
        data_dir: str,
        resume_ckpt: str | None = None,
        batch_size: int = 64,
        image_size: int = 32,
        lr: float = 1e-4,
        max_steps: int = 100000,
        save_interval: int = 5000,
        use_lpips: bool = True,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_steps = max_steps
        self.save_interval = save_interval
        self.use_lpips = use_lpips

        self._setup_ddp()

        # 1. Save the injected models and map to DDP if distributed
        self.online_model = online_model
        self.target_model = target_model
        self.teacher_model = teacher_model

        if self.is_distributed:
            self.online_model = DDP(self.online_model, device_ids=[self.local_rank], output_device=self.local_rank)

        # 2. Optimizer & Data Loader setup
        online_params = self.online_model.module.parameters() if self.is_distributed else self.online_model.parameters()
        self.optimizer = AdamW(online_params, lr=lr)
        self.scaler = GradScaler(self.device.type)

        self.start_step = 0
        if resume_ckpt and os.path.exists(resume_ckpt):
            self._resume_training(resume_ckpt)

        self.data_generator = create_data_loader(
            data_dir=data_dir,
            batch_size=self.batch_size,
            image_size=self.image_size
        )

        if self.use_lpips:
            self.lpips_fn = LPIPS(replace_pooling=True, reduction="none").to(self.device)
        else:
            self.lpips_fn = None

    def _setup_ddp(self):
        # ddp setup logic (same as before, but now in a separate method for clarity)
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
            device=str(self.device)
        )

    def train(self):
        if self.is_main_process:
            print("[*] Starting Consistency Distillation...")

        for step in range(self.start_step, self.max_steps):
            self.online_model.train()
            self.optimizer.zero_grad()

            images, _ = next(self.data_generator)
            images = images.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                loss = consistency_distillation_loss(
                    online_model=self.online_model,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    images=images,
                    num_scales=18,
                    use_lpips=self.use_lpips,
                    lpips_loss_fn=self.lpips_fn
                )

            # AMP backward and optimizer step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            update_ema(self.target_model, self.online_model, mu=0.999)

            # Logging & Saving
            if self.is_main_process and step % 100 == 0:
                print(f"Step {step}/{self.max_steps} | CD Loss: {loss.item():.4f}")

            if self.is_main_process and step > 0 and step % self.save_interval == 0:
                save_checkpoint(
                    save_path=f"checkpoints/step_{step:06d}.pt",
                    step=step,
                    ema_model=self.target_model,
                    model=self.online_model.module if self.is_distributed else self.online_model,
                    optimizer=self.optimizer
                )

        if self.is_distributed:
            dist.destroy_process_group()
