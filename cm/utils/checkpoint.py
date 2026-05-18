import os
import torch
from typing import Any, Dict, Optional, Tuple

def save_checkpoint(
    save_path: str,
    step: int,
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves the training state to a checkpoint file.
    
    Args:
        save_path (str): Path to save the checkpoint file (e.g., 'checkpoints/step_10000.pt').
        step (int): Current training step.
        ema_model (nn.Module): Exponential Moving Average (EMA) model (f_θ⁻).
        model (nn.Module): Online model (f_θ).
        optimizer (optim.Optimizer): Optimizer state.
        config (dict, optional): Hyperparameters or configuration settings used for training.
    """
    
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "ema_state_dict": ema_model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config
    }
    
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path} at step {step}")


def load_checkpoint(
    load_path: str,
    ema_model: torch.nn.Module,                  
    model: Optional[torch.nn.Module] = None,     
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Tuple[int, Optional[Dict[str, Any]]]:  
    """
    Loads a training state from a checkpoint file.
    Can be used for both inference (loading only the EMA model) and resuming training.
    
    Args:
        load_path (str): Path to load the checkpoint file from.
        ema_model (nn.Module): EMA model (f_θ⁻).
        model (nn.Module, optional): Online model (f_θ) - required for resuming training.
        optimizer (optim.Optimizer, optional): Optimizer state - required for resuming training.
        device (str): Device to map the loaded tensors to (e.g., "cpu", "cuda").
        
    Returns:
        Tuple[int, Optional[Dict[str, Any]]]: A tuple containing the step to resume from 
                                              and the config dictionary used during training.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")
        
    checkpoint = torch.load(load_path, map_location=device)
    
    # 1. Load EMA model (f_θ⁻) state
    ema_model.load_state_dict(checkpoint["ema_state_dict"])
    
    # 2. Load online model (f_θ) state (if provided for resuming training)
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        
    # 3. Load optimizer state (if provided for resuming training)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    step = checkpoint.get("step", 0)
    config = checkpoint.get("config", None)
    
    print(f"Loaded checkpoint from {load_path} (step {step})")
    
    return step, config
