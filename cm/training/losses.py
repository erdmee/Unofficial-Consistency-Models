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
    lambda_spectral: float = 0.0,
    spectral_hp_cutoff: float = 0.5,
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

    # HF penalty on native-res preds (before the LPIPS upscale below)
    spectral = (
        snr_adaptive_spectral_loss(
            online_pred, target_pred, t_n_plus_1,
            sigma_data=sigma_data, hp_cutoff=spectral_hp_cutoff,
        )
        if lambda_spectral > 0.0
        else online_pred.new_zeros(())
    )

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

    return loss + lambda_spectral * spectral


def consistency_training_loss(
    online_model: nn.Module,
    target_model: nn.Module,
    images: torch.Tensor,
    num_scales: int,
    sigma_data: float = 0.5,
    use_lpips: bool = False,
    lpips_loss_fn: nn.Module = None,
    lambda_spectral: float = 0.0,
    spectral_hp_cutoff: float = 0.5,
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

    # HF penalty on native-res preds (before the LPIPS upscale below)
    spectral = (
        snr_adaptive_spectral_loss(
            online_pred, target_pred, t_n_plus_1,
            sigma_data=sigma_data, hp_cutoff=spectral_hp_cutoff,
        )
        if lambda_spectral > 0.0
        else online_pred.new_zeros(())
    )

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

    return loss + lambda_spectral * spectral


def snr_adaptive_spectral_loss(
    online_pred: torch.Tensor,
    target_pred: torch.Tensor,
    t: torch.Tensor,
    sigma_data: float = 0.5,
    hp_cutoff: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """SNR-adaptive high-frequency consistency penalty; add to a CD/CT loss.

    HF mismatch between online and (detached) target x0, weighted per-sample by
    w(t) = sigma_data^2 / (t^2 + sigma_data^2) — ~1 at low noise, ~0 at high.
    `t` is the online/high-noise sigma (e.g. t_{n+1}); pass native-res preds.

    Load-bearing: high-pass mask only (full band ~ pixel L2 by Parseval),
    bounded w(t) (not the t^-2 SNR weights), float32 FFT (fp16-unsafe).
    """
    # float32 + ortho norm: fp16-safe and Parseval-consistent (stable lambda)
    online = online_pred.float()
    target = target_pred.float()
    f_online = torch.fft.rfft2(online, norm="ortho")
    f_target = torch.fft.rfft2(target, norm="ortho")

    # high-pass radial mask over the rfft2 grid (H, W//2+1)
    H, W = online.shape[-2], online.shape[-1]
    fy = torch.fft.fftfreq(H, device=online.device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(W, device=online.device, dtype=torch.float32)
    radial = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    mask = (radial >= hp_cutoff * 0.5).to(torch.float32)

    diff = (f_online - f_target) * mask
    power = diff.real ** 2 + diff.imag ** 2

    kept = mask.sum().clamp_min(eps)
    band_power = power.sum(dim=(-2, -1)) / kept             # mean over kept band, (B, C)

    w = sigma_data ** 2 / (t.float() ** 2 + sigma_data ** 2)   # bounded SNR weight, (B,)
    return (band_power.mean(dim=1) * w).mean()