import argparse
from pathlib import Path

import torch
import yaml

from cm.models.unet import UNetModel
from cm.models.precond import ConsistencyPrecond, EDMPrecond
from cm.training.cd_trainer import CDTrainer
from cm.training.ct_trainer import CTTrainer
from cm.diffusion.karras_schedule import ScheduleConfig


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args) -> None:
    """Let CLI flags override the YAML in place; None leaves the config value."""
    if args.max_steps is not None:
        cfg.setdefault("training", {})["max_steps"] = args.max_steps
    if args.lambda_spectral is not None:
        cfg.setdefault("spectral", {})["lambda"] = args.lambda_spectral
    if args.wandb_run_name is not None:
        cfg.setdefault("logging", {})["wandb_run_name"] = args.wandb_run_name


def build_unet(cfg: dict) -> UNetModel:
    m = cfg["model"]
    kwargs = dict(
        image_size=cfg["data"]["image_size"],
        in_channels=cfg["data"].get("channels", 3),
        out_channels=cfg["data"].get("channels", 3),
        model_channels=m["model_channels"],
        channel_mult=tuple(m["channel_mult"]),
        num_res_blocks=m["num_res_blocks"],
        attention_resolutions=tuple(m["attention_resolutions"]),
        dropout=m.get("dropout", 0.0),
        num_classes=m.get("num_classes"),
    )
    # UNetModel accepts either num_heads or num_head_channels (-1 = unused)
    if "num_head_channels" in m:
        kwargs["num_head_channels"] = m["num_head_channels"]
    if "num_heads" in m:
        kwargs["num_heads"] = m["num_heads"]
    return UNetModel(**kwargs)


def build_consistency(cfg: dict, device: torch.device, schedule: ScheduleConfig) -> ConsistencyPrecond:
    """Student/target wrapper with boundary condition f(x, ε)=x enforced (ε = σ_min)."""
    return ConsistencyPrecond(
        build_unet(cfg), sigma_data=schedule.sigma_data, epsilon=schedule.sigma_min
    ).to(device)


def build_edm(cfg: dict, device: torch.device, schedule: ScheduleConfig) -> EDMPrecond:
    """Teacher wrapper — standard EDM preconditioning, no boundary."""
    return EDMPrecond(build_unet(cfg), sigma_data=schedule.sigma_data).to(device)


def load_inner_unet(precond_model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    precond_model.model.load_state_dict(sd, strict=True)


def run_cd(cfg: dict, device: torch.device, resume: str | None) -> None:
    print("[train] mode=cd")
    cd_cfg = cfg["cd"]
    spectral_cfg = cfg.get("spectral", {})
    schedule = ScheduleConfig.from_config(cfg)
    teacher_ckpt = cd_cfg["teacher_ckpt"]
    if not Path(teacher_ckpt).is_file():
        raise FileNotFoundError(
            f"CD requires a teacher checkpoint at {teacher_ckpt}. "
            f"Place a PyTorch state_dict matching our UNet shape there."
        )

    teacher = build_edm(cfg, device, schedule)
    online = build_consistency(cfg, device, schedule)
    target = build_consistency(cfg, device, schedule)
    sampling_ema = build_consistency(cfg, device, schedule)

    print(f"[train] loading teacher state_dict from {teacher_ckpt}")
    load_inner_unet(teacher, teacher_ckpt, device)
    load_inner_unet(online, teacher_ckpt, device)

    # target and sampling EMA start exactly equal to online — paper Sec. 4 / Appx. C.
    target.load_state_dict(online.state_dict())
    sampling_ema.load_state_dict(online.state_dict())

    teacher.requires_grad_(False).eval()
    target.requires_grad_(False).eval()
    sampling_ema.requires_grad_(False).eval()

    trainer = CDTrainer(
        online_model=online,
        target_model=target,
        teacher_model=teacher,
        sampling_ema_model=sampling_ema,
        data_dir=cfg["data"]["data_dir"],
        resume_ckpt=resume,
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        lr=cfg["training"]["lr"],
        max_steps=cfg["training"]["max_steps"],
        save_interval=cfg["logging"]["ckpt_every"],
        use_lpips=(cfg["training"]["loss"] == "lpips"),
        lambda_spectral=spectral_cfg.get("lambda", 0.0),
        spectral_hp_cutoff=spectral_cfg.get("hp_cutoff", 0.5),
        num_scales=cd_cfg["num_scales"],
        target_mu=cd_cfg["target_mu"],
        log_every=cfg["logging"]["log_every"],
        use_fp16=cfg["training"].get("use_fp16", False),
        sampling_ema_decay=cfg["training"].get("sampling_ema_decay", 0.9999),
        class_cond=cfg["model"].get("num_classes") is not None,
        use_wandb=cfg["logging"].get("use_wandb", False),
        wandb_project=cfg["logging"].get("wandb_project", "consistency-models"),
        wandb_run_name=cfg["logging"].get("wandb_run_name"),
        wandb_config={"mode": "cd", **cfg},
        schedule=schedule,
        out_dir=cfg["logging"].get("out_dir", "checkpoints"),
        config=cfg,
    )
    trainer.train()


def run_ct(cfg: dict, device: torch.device, resume: str | None, init_ckpt: str | None = None) -> None:
    print("[train] mode=ct")
    ct_cfg = cfg["ct"]
    spectral_cfg = cfg.get("spectral", {})
    schedule = ScheduleConfig.from_config(cfg)

    online = build_consistency(cfg, device, schedule)
    target = build_consistency(cfg, device, schedule)
    sampling_ema = build_consistency(cfg, device, schedule)

    pretrained = ct_cfg.get("pretrained_ckpt")
    if pretrained:
        pretrained_path = Path(pretrained)
        if pretrained_path.is_file():
            print(f"[train] EDM init from {pretrained}")
            load_inner_unet(online, pretrained, device)
        else:
            print(f"[train] WARNING: pretrained_ckpt={pretrained} not found, falling back to random init")
    else:
        print("[train] random init (no pretrained_ckpt in config)")

    # target and sampling EMA must start exactly equal to online, whether or
    # not we loaded a pretrained UNet — random init otherwise diverges from t=0.
    target.load_state_dict(online.state_dict())
    sampling_ema.load_state_dict(online.state_dict())

    trainer = CTTrainer(
        online_model=online,
        target_model=target,
        sampling_ema_model=sampling_ema,
        data_dir=cfg["data"]["data_dir"],
        resume_ckpt=resume,
        init_ckpt=init_ckpt,
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        lr=cfg["training"]["lr"],
        max_steps=cfg["training"]["max_steps"],
        save_interval=cfg["logging"]["ckpt_every"],
        s0=ct_cfg["s0"],
        s1=ct_cfg["s1"],
        mu0=ct_cfg["mu0"],
        use_lpips=(cfg["training"]["loss"] == "lpips"),
        lambda_spectral=spectral_cfg.get("lambda", 0.0),
        spectral_hp_cutoff=spectral_cfg.get("hp_cutoff", 0.5),
        log_every=cfg["logging"]["log_every"],
        use_fp16=cfg["training"].get("use_fp16", False),
        sampling_ema_decay=cfg["training"].get("sampling_ema_decay", 0.9999),
        class_cond=cfg["model"].get("num_classes") is not None,
        use_wandb=cfg["logging"].get("use_wandb", False),
        wandb_project=cfg["logging"].get("wandb_project", "consistency-models"),
        wandb_run_name=cfg["logging"].get("wandb_run_name"),
        wandb_config={"mode": "ct", **cfg},
        schedule=schedule,
        out_dir=cfg["logging"].get("out_dir", "checkpoints"),
        config=cfg,
    )
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Consistency Models trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", required=True, choices=["cd", "ct"], help="Training mode")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")
    parser.add_argument("--init_ckpt", default=None, help="Warm-start CT weights from a checkpoint (step reset to 0)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument("--lambda_spectral", type=float, default=None, help="Override spectral.lambda")
    parser.add_argument("--wandb_run_name", default=None, help="Override logging.wandb_run_name")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    apply_cli_overrides(cfg, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] config={args.config} mode={args.mode} device={device}")

    if args.mode == "cd":
        run_cd(cfg, device, args.resume)
    else:
        run_ct(cfg, device, args.resume, init_ckpt=args.init_ckpt)


if __name__ == "__main__":
    main()