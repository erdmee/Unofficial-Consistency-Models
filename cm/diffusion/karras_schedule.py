import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScheduleConfig:
    """Single source of truth for the noise-schedule parameters.

    Built from the YAML `schedule:` section via `from_config`; defaults match the
    paper values so configs without that section behave identically.
    Note: the consistency boundary epsilon is taken as sigma_min (t = ε = σ_min),
    so changing sigma_min also moves the boundary condition.
    """
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict) -> "ScheduleConfig":
        sched = cfg.get("schedule") or {}
        return cls(
            sigma_min=float(sched.get("sigma_min", cls.sigma_min)),
            sigma_max=float(sched.get("sigma_max", cls.sigma_max)),
            rho=float(sched.get("rho", cls.rho)),
            sigma_data=float(sched.get("sigma_data", cls.sigma_data)),
        )


def karras_sigmas(
    N: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Karras et al. (2022) noise schedule.

    Returns a 1-D tensor of shape (N,), ordered sigma_min → sigma_max
    (ascending, matches the CM paper convention t_0 = sigma_min, t_n < t_{n+1}).
    """
    ramp = torch.linspace(0, 1, N, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (min_inv_rho + ramp * (max_inv_rho - min_inv_rho)) ** rho
    return sigmas


def n_schedule(
    k: int,
    total_steps: int,
    s0: int,
    s1: int,
) -> int:
    """CT discretization-count schedule N(k) — grows s0 → s1 over training (CM Eq. 11).

    Returns N + 1 to match the paper's "N+1 boundary points" convention.
    """
    progress = min(max(k / total_steps, 0.0), 1.0)
    inner = progress * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2
    n = math.ceil(math.sqrt(inner) - 1)
    return max(n, 1) + 1


def mu_schedule(
    k: int,
    total_steps: int,
    mu0: float,
    s0: int,
    s1: int,
) -> float:
    """CT target-network EMA decay μ(k) = exp(s0 · log(μ0) / N(k)).

    Keeps the effective averaging window roughly constant as N(k) grows.
    By construction μ(0) = μ0 when N(0) = s0.
    """
    n_k = n_schedule(k, total_steps, s0, s1)
    c = -math.log(mu0) * s0
    return math.exp(-c / n_k)
