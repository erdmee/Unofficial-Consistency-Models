import pytest
import torch
import torch.nn as nn

from cm.diffusion.karras_schedule import mu_schedule
from cm.training.ema import update_ema


def _pair() -> tuple[nn.Module, nn.Module]:
    """Online: random init. Target: zeroed so the two are guaranteed different."""
    torch.manual_seed(0)
    online = nn.Linear(4, 4)
    target = nn.Linear(4, 4)
    with torch.no_grad():
        for p in target.parameters():
            p.zero_()
    return online, target


def test_ema_update_with_mu_zero_copies_online_to_target():
    """mu = 0 -> target params equal online params after one update."""
    online, target = _pair()
    update_ema(target, online, mu=0.0)
    for tp, op in zip(target.parameters(), online.parameters()):
        assert torch.equal(tp.data, op.data)


def test_ema_update_with_mu_one_freezes_target():
    """mu = 1 -> target params unchanged after one update."""
    online, target = _pair()
    snapshot = [p.data.clone() for p in target.parameters()]
    update_ema(target, online, mu=1.0)
    for tp, snap in zip(target.parameters(), snapshot):
        assert torch.equal(tp.data, snap)


def test_mu_schedule_within_bounds():
    """mu(k) in (0, 1) for all k in [0, total_steps], monotone non-decreasing."""
    mu0 = 0.95
    s0, s1, total = 2, 150, 1000

    prev = mu_schedule(0, total, mu0, s0, s1)
    for k in range(0, total + 1, 25):
        mu = mu_schedule(k, total, mu0, s0, s1)
        assert 0.0 < mu < 1.0, f"k={k}: mu={mu} outside (0, 1)"
        assert mu >= prev - 1e-12, f"k={k}: mu decreased from {prev} to {mu}"
        prev = mu


def test_mu_schedule_boundary_value():
    """Paper Eq. 12: mu(0) = exp(s0 * log(mu0) / N(0)) = mu0 by construction."""
    mu0 = 0.95
    assert mu_schedule(0, 1000, mu0, 2, 150) == pytest.approx(mu0)
