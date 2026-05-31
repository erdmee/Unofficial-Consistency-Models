"""Shared sampling helpers: rebuild a Consistency Model from a checkpoint and resolve
class conditioning. Used by the onestep and multistep CLIs."""

import torch
import yaml

from cm.models.unet import UNetModel
from cm.models.precond import ConsistencyPrecond


def build_unet_from_config(cfg: dict) -> UNetModel:
    """Build the UNet from a stored config. Mirrors cm.training.train.build_unet
    (kept separate so sampling doesn't import the training stack)."""
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


def load_consistency_model(ckpt_path: str, device: torch.device, config_path: str | None = None):
    """Load a Consistency Model for sampling, rebuilding its architecture from the
    checkpoint's config (or from `config_path` if the checkpoint has none). Prefers the
    sampling EMA weights. Returns (model, config, step)."""
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = checkpoint.get("config")
    if config is None:
        if config_path is None:
            raise ValueError(
                f"Checkpoint '{ckpt_path}' has no embedded 'config' (it was saved with config=None); "
                f"cannot reconstruct the model architecture. Re-run with --config pointing to the "
                f"training YAML for this checkpoint (e.g. --config configs/cifar10_cd.yaml)."
            )
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"[*] Checkpoint has no embedded config; using architecture from '{config_path}'.")

    # Default sigma_data/epsilon to match training (not stored in the state_dict).
    model = ConsistencyPrecond(build_unet_from_config(config)).to(device)

    if "sampling_ema_state_dict" in checkpoint:
        state_dict = checkpoint["sampling_ema_state_dict"]
    else:
        print("Warning: checkpoint has no sampling_ema_state_dict; using target EMA instead.")
        state_dict = checkpoint["ema_state_dict"]
    model.load_state_dict(state_dict)

    step = checkpoint.get("step", 0)
    return model, config, step


def resolve_labels(
    num_classes: int | None,
    batch_size: int,
    class_id: int | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the class-label tensor `y`: `class_id` (or random) when class-conditional,
    None when unconditional (passing `class_id` then is an error)."""
    if num_classes is not None:
        if class_id is not None:
            if not (0 <= class_id < num_classes):
                raise ValueError(
                    f"--class_id={class_id} out of range for num_classes={num_classes} "
                    f"(valid: 0..{num_classes - 1})."
                )
            print(f"[*] Class-conditional sampling with class_id={class_id}.")
            return torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        print(f"[*] Class-conditional sampling with random classes in [0, {num_classes}).")
        return torch.randint(0, num_classes, (batch_size,), dtype=torch.long, device=device)

    if class_id is not None:
        raise ValueError(
            "--class_id was given but this checkpoint is unconditional (config num_classes is null). "
            "Omit --class_id for unconditional models."
        )
    print("[*] Unconditional sampling.")
    return None
