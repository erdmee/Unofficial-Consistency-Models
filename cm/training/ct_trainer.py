import os
from datetime import datetime

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
from cm.diffusion.karras_schedule import ScheduleConfig, n_schedule, mu_schedule


class CTTrainer:
    def __init__(
        self,
        online_model: torch.nn.Module,
        target_model: torch.nn.Module,
        sampling_ema_model: torch.nn.Module,
        data_dir: str,
        resume_ckpt: str | None = None,
        init_ckpt: str | None = None,
        batch_size: int = 512,
        image_size: int = 32,
        lr: float = 2e-4,
        max_steps: int = 800_000,
        save_interval: int = 5000,
        s0: int = 2,
        s1: int = 150,
        mu0: float = 0.95,
        use_lpips: bool = True,
        lambda_spectral: float = 0.0,
        spectral_hp_cutoff: float = 0.5,
        log_every: int = 50,
        use_fp16: bool = False,
        sampling_ema_decay: float = 0.9999,
        class_cond: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "consistency-models",
        wandb_run_name: str | None = None,
        wandb_config: dict | None = None,
        keep_last_steps: int = 100_000,
        keep_milestone_every: int = 100_000,
        schedule: ScheduleConfig | None = None,
        out_dir: str = "checkpoints",
        config: dict | None = None,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_steps = max_steps
        self.save_interval = save_interval
        self.use_lpips = use_lpips
        self.lambda_spectral = lambda_spectral
        self.spectral_hp_cutoff = spectral_hp_cutoff
        self.s0 = s0
        self.s1 = s1
        self.mu0 = mu0
        self.log_every = log_every
        self.use_fp16 = use_fp16
        self.sampling_ema_decay = sampling_ema_decay
        self.class_cond = class_cond
        self.keep_last_steps = keep_last_steps
        self.keep_milestone_every = keep_milestone_every
        self.schedule = schedule or ScheduleConfig()
        self.out_dir = out_dir
        self.config = config

        self._setup_ddp()

        if self.is_main_process:
            s = self.schedule
            print(
                f"[train] schedule: sigma_min={s.sigma_min} sigma_max={s.sigma_max} "
                f"rho={s.rho} sigma_data={s.sigma_data} | ckpt_dir={self.out_dir}"
            )

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
        self.wandb_run_id: str | None = None
        if resume_ckpt and os.path.exists(resume_ckpt):
            self._resume_training(resume_ckpt)
        elif init_ckpt and os.path.exists(init_ckpt):
            self._init_from_checkpoint(init_ckpt)

        self.data_generator = create_data_loader(
            data_dir=data_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            class_cond=self.class_cond,
        )

        if self.use_lpips:
            self.lpips_fn = LPIPS(replace_pooling=True, reduction="none").to(self.device)
        else:
            self.lpips_fn = None

        self.use_wandb = use_wandb and self.is_main_process
        if self.use_wandb:
            import wandb
            self._wandb = wandb
            if self.wandb_run_id:
                wandb.init(
                    project=wandb_project,
                    id=self.wandb_run_id,
                    resume="allow",
                    config=wandb_config or {},
                )
            else:
                run_name = wandb_run_name or f"ct_{datetime.now():%Y%m%d_%H%M%S}"
                wandb.init(project=wandb_project, name=run_name, config=wandb_config or {})
            self.wandb_run_id = self._wandb.run.id
        else:
            self._wandb = None

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
        loaded_step, _, self.wandb_run_id = load_checkpoint(
            load_path=resume_ckpt,
            ema_model=self.target_model,
            model=self.online_model.module if self.is_distributed else self.online_model,
            optimizer=self.optimizer,
            sampling_ema_model=self.sampling_ema_model,
            device=str(self.device),
            scaler=self.scaler,
        )
        self.start_step = loaded_step + 1

    def _init_from_checkpoint(self, init_ckpt: str):
        """Warm-start weights from a checkpoint, keeping step=0 and a fresh optimizer/wandb run."""
        if self.is_main_process:
            print(f"[*] Warm-starting weights from {init_ckpt} (step reset to 0, fresh optimizer)...")
        load_checkpoint(
            load_path=init_ckpt,
            ema_model=self.target_model,
            model=self.online_model.module if self.is_distributed else self.online_model,
            optimizer=None,
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

            images, cond = next(self.data_generator)
            images = images.to(self.device)
            y = cond["y"].to(self.device) if self.class_cond else None

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_fp16):
                loss = consistency_training_loss(
                    online_model=self.online_model,
                    target_model=self.target_model,
                    images=images,
                    num_scales=N_k,
                    schedule=self.schedule,
                    use_lpips=self.use_lpips,
                    lpips_loss_fn=self.lpips_fn,
                    lambda_spectral=self.lambda_spectral,
                    spectral_hp_cutoff=self.spectral_hp_cutoff,
                    y=y,
                )

            self.scaler.scale(loss).backward()

            grad_norm = None
            param_norm = None
            if self.is_main_process and step % self.log_every == 0:
                with torch.no_grad():
                    params = list(self.online_model.parameters())
                    param_norm = torch.stack([p.norm(2) for p in params]).norm(2).item()
                    grad_norms = torch.stack([p.grad.norm(2) for p in params if p.grad is not None])
                    grad_scale = self.scaler.get_scale() if self.use_fp16 else 1.0
                    grad_norm = (grad_norms.norm(2) / grad_scale).item()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            update_ema(self.target_model, self.online_model, mu=mu_k)
            update_ema(self.sampling_ema_model, self.online_model, mu=self.sampling_ema_decay)

            if self.is_main_process and step % self.log_every == 0:
                print(
                    f"Step {step}/{self.max_steps} | "
                    f"N(k)={N_k} | μ(k)={mu_k:.5f} | "
                    f"CT Loss: {loss.item():.4f} | "
                    f"grad_norm={grad_norm:.3f} | param_norm={param_norm:.3f}"
                )
                if self.use_wandb:
                    self._wandb.log(
                        {
                            "loss": loss.item(),
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "N_k": N_k,
                            "mu_k": mu_k,
                            "grad_norm": grad_norm,
                            "param_norm": param_norm,
                        },
                        step=step,
                    )

            if self.is_main_process and step > 0 and step % self.save_interval == 0:
                save_checkpoint(
                    save_path=os.path.join(self.out_dir, f"step_{step:06d}.pt"),
                    step=step,
                    ema_model=self.target_model,
                    model=self.online_model.module if self.is_distributed else self.online_model,
                    optimizer=self.optimizer,
                    sampling_ema_model=self.sampling_ema_model,
                    config=self.config,
                    keep_last_steps=self.keep_last_steps,
                    keep_milestone_every=self.keep_milestone_every,
                    wandb_run_id=self.wandb_run_id,
                    scaler=self.scaler,
                )

        if self.is_main_process and self.max_steps > 0:
            final_step = self.max_steps - 1
            save_checkpoint(
                save_path=os.path.join(self.out_dir, f"step_{final_step:06d}.pt"),
                step=final_step,
                ema_model=self.target_model,
                model=self.online_model.module if self.is_distributed else self.online_model,
                optimizer=self.optimizer,
                sampling_ema_model=self.sampling_ema_model,
                config=self.config,
                keep_last_steps=self.keep_last_steps,
                keep_milestone_every=self.keep_milestone_every,
                wandb_run_id=self.wandb_run_id,
                scaler=self.scaler,
            )

        if self.use_wandb:
            self._wandb.finish()

        if self.is_distributed:
            dist.destroy_process_group()
