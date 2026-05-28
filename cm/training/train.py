import argparse
from pathlib import Path

import torch
import yaml

from cm.models.unet import UNetModel
from cm.models.precond import ConsistencyPrecond, EDMPrecond
from cm.training.cd_trainer import CDTrainer
from cm.training.ct_trainer import CTTrainer


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def build_consistency(cfg: dict, device: torch.device) -> ConsistencyPrecond:
    """Student/target wrapper with boundary condition f(x, ε)=x enforced."""
    return ConsistencyPrecond(build_unet(cfg)).to(device)


def build_edm(cfg: dict, device: torch.device) -> EDMPrecond:
    """Teacher wrapper — standard EDM preconditioning, no boundary."""
    return EDMPrecond(build_unet(cfg)).to(device)


def load_inner_unet(precond_model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    precond_model.model.load_state_dict(sd, strict=True)


def run_cd(cfg: dict, device: torch.device, resume: str | None) -> None:
    print("[train] mode=cd")
    cd_cfg = cfg["cd"]
    teacher_ckpt = cd_cfg["teacher_ckpt"]
    if not Path(teacher_ckpt).is_file():
        raise FileNotFoundError(
            f"CD requires a teacher checkpoint at {teacher_ckpt}. "
            f"Place a PyTorch state_dict matching our UNet shape there."
        )

    teacher = build_edm(cfg, device)
    online = build_consistency(cfg, device)
    target = build_consistency(cfg, device)

    print(f"[train] loading teacher state_dict from {teacher_ckpt}")
    load_inner_unet(teacher, teacher_ckpt, device)
    load_inner_unet(online, teacher_ckpt, device)
    load_inner_unet(target, teacher_ckpt, device)

    teacher.requires_grad_(False).eval()
    target.requires_grad_(False).eval()

    trainer = CDTrainer(
        online_model=online,
        target_model=target,
        teacher_model=teacher,
        data_dir=cfg["data"]["data_dir"],
        resume_ckpt=resume,
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        lr=cfg["training"]["lr"],
        max_steps=cfg["training"]["max_steps"],
        save_interval=cfg["logging"]["ckpt_every"],
        use_lpips=(cfg["training"]["loss"] == "lpips"),
        num_scales=cd_cfg["num_scales"],
        target_mu=cd_cfg["target_mu"],
        log_every=cfg["logging"]["log_every"],
    )
    trainer.train()


def run_ct(cfg: dict, device: torch.device, resume: str | None) -> None:
    print("[train] mode=ct")
    ct_cfg = cfg["ct"]

    online = build_consistency(cfg, device)
    target = build_consistency(cfg, device)

    pretrained = ct_cfg.get("pretrained_ckpt")
    if pretrained:
        pretrained_path = Path(pretrained)
        if pretrained_path.is_file():
            print(f"[train] EDM init from {pretrained}")
            load_inner_unet(online, pretrained, device)
            load_inner_unet(target, pretrained, device)
        else:
            print(f"[train] WARNING: pretrained_ckpt={pretrained} not found, falling back to random init")
    else:
        print("[train] random init (no pretrained_ckpt in config)")

    trainer = CTTrainer(
        online_model=online,
        target_model=target,
        data_dir=cfg["data"]["data_dir"],
        resume_ckpt=resume,
        batch_size=cfg["training"]["batch_size"],
        image_size=cfg["data"]["image_size"],
        lr=cfg["training"]["lr"],
        max_steps=cfg["training"]["max_steps"],
        save_interval=cfg["logging"]["ckpt_every"],
        s0=ct_cfg["s0"],
        s1=ct_cfg["s1"],
        mu0=ct_cfg["mu0"],
        use_lpips=(cfg["training"]["loss"] == "lpips"),
        log_every=cfg["logging"]["log_every"],
    )
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Consistency Models trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", required=True, choices=["cd", "ct"], help="Training mode")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] config={args.config} mode={args.mode} device={device}")

    if args.mode == "cd":
        run_cd(cfg, device, args.resume)
    else:
        run_ct(cfg, device, args.resume)


if __name__ == "__main__":
    main()