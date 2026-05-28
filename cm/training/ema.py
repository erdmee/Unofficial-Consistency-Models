import torch
import torch.nn as nn


@torch.no_grad()
def update_ema(target_model: nn.Module, online_model: nn.Module, mu: float = 0.999):
    """EMA update: θ_target ← μ · θ_target + (1 − μ) · θ_online. Unwraps DDP `.module` if present."""
    online_params = online_model.module.parameters() if hasattr(online_model, "module") else online_model.parameters()
    target_params = target_model.module.parameters() if hasattr(target_model, "module") else target_model.parameters()

    for target_param, online_param in zip(target_params, online_params):
        target_param.data.mul_(mu).add_(online_param.data, alpha=1.0 - mu)


def get_adaptive_ema_rate(step: int, max_steps: int, start_mu: float = 0.95, end_mu: float = 0.999) -> float:
    """Linear EMA ramp from start_mu to end_mu over training. Unused by the default CT/CD trainers."""
    progress = min(1.0, step / max_steps)
    return start_mu + (end_mu - start_mu) * progress
