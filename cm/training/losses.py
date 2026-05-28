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
    lpips_loss_fn: nn.Module = None,
    y: torch.Tensor | None = None,
):
    """CD loss with a single Heun step from teacher.

    Paper indexing: t_n < t_{n+1}; online sees t_{n+1}, target sees t_n.
    `y` is the per-sample class label tensor for class-conditional models (None for unconditional).
    """
    device = images.device
    batch_size = images.shape[0]

    sigmas = karras_sigmas(num_scales, device=device)

    indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)
    t_n        = sigmas[indices]
    t_n_plus_1 = sigmas[indices + 1]

    noise = torch.randn_like(images)
    x_t_n_plus_1 = images + noise * t_n_plus_1.view(-1, 1, 1, 1)

    x_t_n = heun_solver(teacher_model, x_t_n_plus_1, t_n_plus_1, t_n, y=y).detach()

    online_pred = online_model(x_t_n_plus_1, t_n_plus_1, y)

    with torch.no_grad():
        target_pred = target_model(x_t_n, t_n, y).detach()

    # SNR + data-variance weighting anchored to the high-noise side
    snrs = t_n_plus_1 ** -2
    weights = (snrs + (1.0 / sigma_data ** 2)).view(-1, 1, 1, 1)

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


def consistency_training_loss(
    online_model: nn.Module,
    target_model: nn.Module,
    images: torch.Tensor,
    num_scales: int,
    sigma_data: float = 0.5,
    use_lpips: bool = False,
    lpips_loss_fn: nn.Module = None,
    y: torch.Tensor | None = None,
):
    """CT loss — no teacher.

    Pairs (x_{t_n}, x_{t_{n+1}}) share a single noise z, which is the CT trick
    that removes the teacher (Song et al. 2023, Algorithm 3).
    num_scales = N(k) is supplied per training step by the trainer.
    `y` is the per-sample class label tensor for class-conditional models (None for unconditional).
    """
    device = images.device
    batch_size = images.shape[0]

    sigmas = karras_sigmas(num_scales, device=device)

    indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)
    t_n        = sigmas[indices]
    t_n_plus_1 = sigmas[indices + 1]

    z = torch.randn_like(images)
    x_t_n        = images + z * t_n.view(-1, 1, 1, 1)
    x_t_n_plus_1 = images + z * t_n_plus_1.view(-1, 1, 1, 1)

    online_pred = online_model(x_t_n_plus_1, t_n_plus_1, y)

    with torch.no_grad():
        target_pred = target_model(x_t_n, t_n, y).detach()

    snrs = t_n_plus_1 ** -2
    weights = (snrs + (1.0 / sigma_data ** 2)).view(-1, 1, 1, 1)

    if use_lpips and lpips_loss_fn is not None:
        if online_pred.shape[-1] < 256:
            online_pred = F.interpolate(online_pred, size=224, mode="bilinear")
            target_pred = F.interpolate(target_pred, size=224, mode="bilinear")

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