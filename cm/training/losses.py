import torch
import torch.nn as nn
import torch.nn.functional as F

from cm.diffusion.solvers import heun_solver
from cm.diffusion.karras_schedule import karras_sigmas


def consistency_distillation_loss(
    online_model: nn.Module,
    target_model: nn.Module,
    teacher_model: nn.Module,
    images: torch.Tensor,
    num_scales: int = 18,
    sigma_data: float = 0.5,
    use_lpips: bool = False,
    lpips_loss_fn: nn.Module = None
):
    """
    Calculates the Consistency Distillation (CD) loss for a given batch of images.
    Pairs (x_{t_n}, x_{t_{n+1}}) where x_{t_n} is obtained by a single Heun step.
    Paper indexing: t_n < t_{n+1}; online sees t_{n+1}, target sees t_n.
    """
    device = images.device
    batch_size = images.shape[0]

    # 1. Current step's σ schedule (ascending: sigmas[0] = sigma_min)
    sigmas = karras_sigmas(num_scales, device=device)

    # 2. Sample n ~ Uniform{0, ..., N-2}; pair (t_n, t_{n+1}) with t_n < t_{n+1}
    indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)
    t_n        = sigmas[indices]
    t_n_plus_1 = sigmas[indices + 1]

    # 3. Add noise to clean images at the higher noise level t_{n+1}
    noise = torch.randn_like(images)
    x_t_n_plus_1 = images + noise * t_n_plus_1.view(-1, 1, 1, 1)

    # 4. Teacher's Heun step: predict x_{t_n} from x_{t_{n+1}}
    x_t_n = heun_solver(teacher_model, x_t_n_plus_1, t_n_plus_1, t_n).detach()

    # 5. Online (high noise) prediction — receives gradient
    online_pred = online_model(x_t_n_plus_1, t_n_plus_1)

    # 6. Target (low noise) prediction — EMA copy, stop gradient
    with torch.no_grad():
        target_pred = target_model(x_t_n, t_n).detach()

    # 7. SNR + data-variance weighting (anchored to the high-noise side)
    snrs = t_n_plus_1 ** -2
    weights = (snrs + (1.0 / sigma_data ** 2)).view(-1, 1, 1, 1)

    # 8. Distance metric d(·,·)
    if use_lpips and lpips_loss_fn is not None:
        # Upscale to 224 before LPIPS — VGG backbone expects ~ImageNet scale
        if online_pred.shape[-1] < 256:
            online_pred = F.interpolate(online_pred, size=224, mode="bilinear")
            target_pred = F.interpolate(target_pred, size=224, mode="bilinear")

        # piq.LPIPS(reduction="none") returns (B,); weights.squeeze() → (B,)
        loss = (
            lpips_loss_fn(
                (online_pred + 1) / 2.0,
                (target_pred + 1) / 2.0,
            ) * weights.squeeze()
        ).mean()
    else:
        # MSE (L2) Loss
        raw_loss = (online_pred - target_pred) ** 2
        loss = (raw_loss * weights).mean()

    return loss


def consistency_training_loss(
    online_model: nn.Module,
    target_model: nn.Module,
    images: torch.Tensor,
    num_scales: int,
    sigma_data: float = 0.5,
    use_lpips: bool = False,
    lpips_loss_fn: nn.Module = None
):
    """
    Calculates the Consistency Training (CT) loss — no teacher network.
    Pairs (x_{t_n}, x_{t_{n+1}}) sharing the same noise z, which removes the teacher requirement (Song et al. 2023, Algorithm 3).
    num_scales = N(k) is expected to be supplied per training step by the trainer.
    Paper indexing: t_n < t_{n+1}; online sees t_{n+1}, target sees t_n.
    """
    device = images.device
    batch_size = images.shape[0]

    # 1. Current step's σ schedule (ascending: sigmas[0] = sigma_min)
    sigmas = karras_sigmas(num_scales, device=device)

    # 2. Sample n ~ Uniform{0, ..., N-2}; pair (t_n, t_{n+1}) with t_n < t_{n+1}
    indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)
    t_n        = sigmas[indices]
    t_n_plus_1 = sigmas[indices + 1]

    # 3. Build the noisy pair from a single shared z (the CT trick — no teacher)
    z = torch.randn_like(images)
    x_t_n        = images + z * t_n.view(-1, 1, 1, 1)
    x_t_n_plus_1 = images + z * t_n_plus_1.view(-1, 1, 1, 1)

    # 4. Online (high noise) prediction — receives gradient
    online_pred = online_model(x_t_n_plus_1, t_n_plus_1)

    # 5. Target (low noise) prediction — EMA copy, stop gradient
    with torch.no_grad():
        target_pred = target_model(x_t_n, t_n).detach()

    # 6. SNR + data-variance weighting (anchored to the high-noise side)
    snrs = t_n_plus_1 ** -2
    weights = (snrs + (1.0 / sigma_data ** 2)).view(-1, 1, 1, 1)

    # 7. Distance metric d(·,·)
    if use_lpips and lpips_loss_fn is not None:
        # Upscale to 224 before LPIPS — VGG backbone expects ~ImageNet scale
        if online_pred.shape[-1] < 256:
            online_pred = F.interpolate(online_pred, size=224, mode="bilinear")
            target_pred = F.interpolate(target_pred, size=224, mode="bilinear")

        # piq.LPIPS(reduction="none") returns (B,); weights.squeeze() → (B,)
        loss = (
            lpips_loss_fn(
                (online_pred + 1) / 2.0,
                (target_pred + 1) / 2.0,
            ) * weights.squeeze()
        ).mean()
    else:
        raw_loss = (online_pred - target_pred) ** 2
        loss = (raw_loss * weights).mean()

    return loss
