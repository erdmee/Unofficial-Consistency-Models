import pytest


@pytest.mark.skip(reason="implement after cm.training.ema")
def test_ema_update_with_mu_zero_copies_online_to_target():
    """mu = 0 -> target params equal online params after one update."""


@pytest.mark.skip(reason="implement after cm.training.ema")
def test_ema_update_with_mu_one_freezes_target():
    """mu = 1 -> target params unchanged after one update."""


@pytest.mark.skip(reason="implement after cm.diffusion.karras_schedule")
def test_mu_schedule_within_bounds():
    """mu(k) in [mu0, 1) for all k in [0, total_steps]."""
