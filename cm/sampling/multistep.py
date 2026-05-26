import math
import torch
import torch.nn as nn


@torch.no_grad()
def generate_multistep(
    model: nn.Module,
    batch_size: int,
    image_size: int,
    device: torch.device,
    ts: list[float],
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """
    Multistep generation from a trained Consistency Model.

    Args:
        model (nn.Module): Trained consistency model.
        batch_size (int): Number of images to generate.
        image_size (int): Spatial resolution H = W.
        device (torch.device): Target device.
        ts (list[float]): Descending intermediate noise levels τ_1 > τ_2 > ... > τ_M,
            each in the open interval (sigma_min, sigma_max).
            Paper-recommended CIFAR-10 schedules:
              NFE = 2 → ts = [0.821]
              NFE = 4 → ts = [24.4, 5.84, 0.9]
        sigma_min (float): Smallest noise level (boundary, paper default 0.002).
        sigma_max (float): Largest noise level (initial noise, paper default 80.0).

    Returns:
        torch.Tensor: Generated images clamped to [-1, 1], shape (B, 3, H, W).
    """
    model.eval()
    shape = (batch_size, 3, image_size, image_size)

    # 1. Initial pure noise at σ_max
    x = torch.randn(*shape, device=device) * sigma_max

    # 2. First denoise from σ_max — single-step's output is this line alone
    t_init = torch.full((batch_size,), sigma_max, device=device)
    x = model(x, t_init)

    # 3. Iterative refinement at descending τ levels
    for tau in ts:
        if not (sigma_min < tau < sigma_max):
            raise ValueError(
                f"tau={tau} must lie strictly between sigma_min={sigma_min} "
                f"and sigma_max={sigma_max}"
            )

        # Re-noise: x_τ = x_0_hat + sqrt(τ² - σ_min²) · z
        # The sqrt(τ² - σ_min²) term accounts for the boundary condition
        # f_θ(x, σ_min) = x, so x_0_hat is treated as living at noise level σ_min.
        z = torch.randn_like(x)
        x = x + z * math.sqrt(tau ** 2 - sigma_min ** 2)

        # Denoise at this τ
        t = torch.full((batch_size,), tau, device=device)
        x = model(x, t)

    # 4. Clamp to valid image range
    return torch.clamp(x, -1.0, 1.0)