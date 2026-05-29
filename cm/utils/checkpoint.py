import os
import re
from typing import Any, Dict, Optional, Tuple

import torch


_CKPT_FILENAME_RE = re.compile(r"^step_(\d+)\.pt$")


def cleanup_old_checkpoints(
    dir_path: str,
    current_step: int,
    keep_last_steps: int,
    keep_milestone_every: int = 0,
) -> None:
    """Delete step_*.pt files older than `current_step - keep_last_steps`.

    Files whose step is a positive multiple of `keep_milestone_every` are protected
    (e.g. 100k, 200k, ... — kept indefinitely for cross-run step alignment).
    """
    if not os.path.isdir(dir_path):
        return
    threshold = current_step - keep_last_steps
    for filename in os.listdir(dir_path):
        m = _CKPT_FILENAME_RE.match(filename)
        if m is None:
            continue
        file_step = int(m.group(1))
        if file_step >= threshold:
            continue
        if keep_milestone_every > 0 and file_step > 0 and file_step % keep_milestone_every == 0:
            continue
        try:
            os.remove(os.path.join(dir_path, filename))
            print(f"Removed old checkpoint {filename}")
        except OSError as e:
            print(f"Failed to remove {filename}: {e}")


def save_checkpoint(
    save_path: str,
    step: int,
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampling_ema_model: Optional[torch.nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    keep_last_steps: Optional[int] = None,
    keep_milestone_every: int = 0,
    wandb_run_id: Optional[str] = None,
) -> None:
    """Save target EMA + online + optimizer (+ optional sampling EMA) to a single .pt file.

    If `keep_last_steps` is set, deletes sibling step_*.pt files whose step is older
    than `step - keep_last_steps`. Files at positive multiples of `keep_milestone_every`
    are protected from deletion (kept for cross-run step alignment).

    `wandb_run_id`, when provided, is stored so the run can be resumed into the same
    wandb dashboard run on a subsequent `load_checkpoint`.
    """
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    checkpoint = {
        "step": step,
        "ema_state_dict": ema_model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "wandb_run_id": wandb_run_id,
    }
    if sampling_ema_model is not None:
        checkpoint["sampling_ema_state_dict"] = sampling_ema_model.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path} at step {step}")

    if keep_last_steps is not None and dir_name:
        cleanup_old_checkpoints(dir_name, step, keep_last_steps, keep_milestone_every)


def load_checkpoint(
    load_path: str,
    ema_model: torch.nn.Module,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    sampling_ema_model: Optional[torch.nn.Module] = None,
    device: str = "cpu",
) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
    """Load checkpoint. For inference only the EMA model is required; pass `model`/`optimizer` to resume training.

    Returns (step, config, wandb_run_id). `wandb_run_id` is None for legacy checkpoints
    saved before that field was added.
    """
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
    wandb_run_id = checkpoint.get("wandb_run_id", None)

    print(f"Loaded checkpoint from {load_path} (step {step})")

    return step, config, wandb_run_id
