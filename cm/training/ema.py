import torch
import torch.nn as nn

@torch.no_grad()
def update_ema(target_model: nn.Module, online_model: nn.Module, mu: float = 0.999):
    """
    Updates the target model parameters using Exponential Moving Average (EMA).
    Target parameters are updated as: theta_target = mu * theta_target + (1 - mu) * theta_online
    
    Args:
        target_model (nn.Module): The model to be updated (f_θ-).
        online_model (nn.Module): The actively training model (f_θ).
        mu (float): The EMA decay rate.
    """
    # In distributed training (DDP) environments, the online_model may be wrapped in 'module', so we unwrap it.
    online_params = online_model.module.parameters() if hasattr(online_model, 'module') else online_model.parameters()
    target_params = target_model.module.parameters() if hasattr(target_model, 'module') else target_model.parameters()

    for target_param, online_param in zip(target_params, online_params):
        target_param.data.mul_(mu).add_(online_param.data, alpha=1.0 - mu)

def get_adaptive_ema_rate(step: int, max_steps: int, start_mu: float = 0.95, end_mu: float = 0.999) -> float:
    """
    Optional: Calculates an adaptive EMA rate that starts lower and approaches end_mu.
    Helps the target model track the online model faster in the early stages of training.
    """
    progress = min(1.0, step / max_steps)
    # Linear interpolation (can be changed to cosine/log scale if needed)
    return start_mu + (end_mu - start_mu) * progress