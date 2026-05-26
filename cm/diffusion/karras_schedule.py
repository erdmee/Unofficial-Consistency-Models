import math
import torch

def karras_sigmas(
    N: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Builds the discretized noise schedule from Karras et al. (2022).
    Returns: torch.Tensor: 1-D tensor of shape (N,), ordered sigma_min -> sigma_max
    (ascending matches the CM paper convention t_0 = sigma_min, t_n < t_{n+1}).
    """
    # 1. Uniform ramp in [0, 1] on the inverted axis
    ramp = torch.linspace(0, 1, N, device=device)

    # 2. Endpoints in the rho-inverted space
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)

    # 3. Linear interpolation in the inverted axis, then power back
    sigmas = (min_inv_rho + ramp * (max_inv_rho - min_inv_rho)) ** rho

    return sigmas


def n_schedule(
    k: int,
    total_steps: int,
    s0: int,
    s1: int,
) -> int:
    """
    CT discretization-count schedule N(k).
    Grows the number of timestep buckets sub-linearly from s0 to s1 over training.
    Reduces variance early (few buckets) and tightens the consistency constraint later (many buckets).
    """
    # 1. Clamp progress to [0, 1] so callers passing k > total_steps don't blow up
    progress = min(max(k / total_steps, 0.0), 1.0)

    # 2. Quadratic in progress growth on the (N+1)^2 axis
    inner = progress * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2

    # 3. Invert via sqrt ceil so we never undershoot the bucket count
    n = math.ceil(math.sqrt(inner) - 1)

    # 4. +1 to match paper's "N+1 boundary points" convention
    return max(n, 1) + 1


def mu_schedule(
    k: int,
    total_steps: int,
    mu0: float,
    s0: int,
    s1: int,
) -> float:
    """
    CT target-network EMA decay schedule μ(k).
    Keeps the effective EMA averaging window roughly constant as
    N(k) grows: μ(k) = exp(s0 * log(μ0) / N(k)). When k = 0, N(k) = s0 and μ(k) = μ0 by construction.
    """
    # 1. Paper's N(k) — n_schedule already returns this (CM Eq. 11).
    n_k = n_schedule(k, total_steps, s0, s1)

    # 2. Solve c such that μ(0) == mu0 when N(0) == s0
    c = -math.log(mu0) * s0

    # 3. Exponentially decay as buckets grow
    return math.exp(-c / n_k)