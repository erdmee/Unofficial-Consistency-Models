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
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-step consistency sampling.

    `ts` is the descending list of intermediate noise levels τ_1 > τ_2 > ... > τ_M,
    each strictly in (sigma_min, sigma_max). Paper-recommended CIFAR-10 values:
        NFE = 2 → ts = [0.821]
        NFE = 4 → ts = [24.4, 5.84, 0.9]
    `y` is the per-sample class label tensor for class-conditional models (None for unconditional).
    Returned images are clamped to [-1, 1].
    """
    model.eval()
    shape = (batch_size, 3, image_size, image_size)

    x = torch.randn(*shape, device=device) * sigma_max

    t_init = torch.full((batch_size,), sigma_max, device=device)
    x = model(x, t_init, y)

    for tau in ts:
        if not (sigma_min < tau < sigma_max):
            raise ValueError(
                f"tau={tau} must lie strictly between sigma_min={sigma_min} "
                f"and sigma_max={sigma_max}"
            )

        z = torch.randn_like(x)
        x = x + z * math.sqrt(tau ** 2 - sigma_min ** 2)

        t = torch.full((batch_size,), tau, device=device)
        x = model(x, t, y)

    return torch.clamp(x, -1.0, 1.0)
