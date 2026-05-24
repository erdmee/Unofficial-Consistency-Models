import torch
import torch.nn as nn

from cm.diffusion.solvers import heun_solver

def consistency_distillation_loss(
    online_model: nn.Module,
    target_model: nn.Module,
    teacher_model: nn.Module,
    images: torch.Tensor,
    num_scales: int = 18,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    rho: float = 7.0,
    sigma_data: float = 0.5,
    use_lpips: bool = False,
    lpips_loss_fn: nn.Module = None
):
    """
    Calculates the Consistency Distillation (CD) loss for a given batch of images.
    """
    device = images.device
    batch_size = images.shape[0]

    # 1. Sample discrete time indices
    indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)

    # 2. Calculate t_n and t_{n-1} based on the Karras schedule
    t_n = (sigma_max ** (1 / rho) + indices / (num_scales - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )) ** rho

    t_n-1 = (sigma_max ** (1 / rho) + (indices + 1) / (num_scales - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )) ** rho

    # 3. Add noise to clean images
    noise = torch.randn_like(images)
    x_t_n = images + noise * t_n.view(-1, 1, 1, 1)

    # 4. Teacher's Heun Step: Predict x_{t_{n-1}} from x_{t_n}
    x_t_n-1 = heun_solver(teacher_model, x_t_n, t_n, t_n-1).detach()

    # 5. Online Model Prediction (Distiller)
    online_pred = online_model(x_t_n, t_n)

    # 6. Target Model Prediction (EMA Distiller)
    with torch.no_grad():
        target_pred = target_model(x_t_n-1, t_n-1).detach()

    # 7. Calculate Weights (SNR + Karras weight schedule)
    snrs = t_n ** -2
    weights = snrs + (1.0 / sigma_data ** 2)
    weights = weights.view(-1, 1, 1, 1)

    # 8. Compute Final Loss
    if use_lpips and lpips_loss_fn is not None:
        # LPIPS expects inputs in [0, 1] range typically
        online_norm = (online_pred + 1.0) / 2.0
        target_norm = (target_pred + 1.0) / 2.0
        
        raw_loss = lpips_loss_fn(online_norm, target_norm)
        # lpips output might be (B, 1, 1, 1) or (B,), adapt accordingly
        loss = (raw_loss.view(batch_size, -1).mean(dim=1) * weights.squeeze()).mean()
    else:
        # MSE (L2) Loss
        raw_loss = (online_pred - target_pred) ** 2
        loss = (raw_loss * weights).mean()

    return loss