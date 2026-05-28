import os
from typing import Any, Dict, Optional, Tuple

import torch


def save_checkpoint(
    save_path: str,
    step: int,
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampling_ema_model: Optional[torch.nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save target EMA + online + optimizer (+ optional sampling EMA) to a single .pt file."""
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    checkpoint = {
        "step": step,
        "ema_state_dict": ema_model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if sampling_ema_model is not None:
        checkpoint["sampling_ema_state_dict"] = sampling_ema_model.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path} at step {step}")


def load_checkpoint(
    load_path: str,
    ema_model: torch.nn.Module,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    sampling_ema_model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Load checkpoint. For inference only the EMA model is required; pass `model`/`optimizer` to resume training."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")

    checkpoint = torch.load(load_path, map_location=device)

    ema_model.load_state_dict(checkpoint["ema_state_dict"])

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if sampling_ema_model is not None:
        if "sampling_ema_state_dict" in checkpoint:
            sampling_ema_model.load_state_dict(checkpoint["sampling_ema_state_dict"])
        else:
            # Legacy checkpoint — fall back to target EMA so resume still works.
            print("Warning: checkpoint has no sampling_ema_state_dict; copying target EMA into sampling EMA.")
            sampling_ema_model.load_state_dict(checkpoint["ema_state_dict"])

    step = checkpoint.get("step", 0)
    config = checkpoint.get("config", None)

    print(f"Loaded checkpoint from {load_path} (step {step})")

    return step, config
