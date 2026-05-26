from typing import Tuple

import torch


def compute_statistics(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (mean, covariance) of a feature set.

    Args:
        features: (N, D) tensor of activations.
    Returns:
        mu: (D,)
        sigma: (D, D)
    """
    if features.dim() != 2:
        raise ValueError(f"expected (N, D), got {tuple(features.shape)}")
    if features.size(0) < 2:
        raise ValueError("need at least 2 samples to estimate covariance")

    features = features.double()
    mu = features.mean(dim=0)
    centered = features - mu
    n = features.size(0)
    sigma = centered.t() @ centered / (n - 1)
    return mu, sigma


def _matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Symmetric PSD matrix square root via eigendecomposition."""
    # Symmetrize to fight numerical drift.
    mat = (mat + mat.t()) / 2.0
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp(min=eps)
    return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.t()


def frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute FID = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 sqrt(S1 S2)).

    Uses the reformulation:
        Tr(sqrtm(S1 S2)) = Tr(sqrtm(A S2 A))   where A = sqrtm(S1),
    which keeps everything inside symmetric PSD eigendecompositions and avoids
    needing scipy.
    """
    mu1 = mu1.double()
    mu2 = mu2.double()
    sigma1 = sigma1.double()
    sigma2 = sigma2.double()

    diff = mu1 - mu2
    mean_term = diff @ diff

    # Add a tiny ridge for numerical PSD-ness.
    d = sigma1.shape[0]
    offset = eps * torch.eye(d, dtype=sigma1.dtype, device=sigma1.device)
    sigma1 = sigma1 + offset
    sigma2 = sigma2 + offset

    a = _matrix_sqrt_psd(sigma1)
    inner = a @ sigma2 @ a
    inner = (inner + inner.t()) / 2.0
    eigvals = torch.linalg.eigvalsh(inner).clamp(min=0.0)
    tr_sqrt = eigvals.sqrt().sum()

    fid = mean_term + torch.trace(sigma1) + torch.trace(sigma2) - 2.0 * tr_sqrt
    return fid


def compute_fid(features_real: torch.Tensor, features_fake: torch.Tensor) -> float:
    """
    End-to-end FID from two (N, D) feature tensors.

    Returns a Python float so it's directly loggable.
    """
    mu_r, sigma_r = compute_statistics(features_real)
    mu_f, sigma_f = compute_statistics(features_fake)
    return frechet_distance(mu_r, sigma_r, mu_f, sigma_f).item()
