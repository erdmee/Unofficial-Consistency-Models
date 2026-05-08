import pytest


@pytest.mark.skip(reason="implement after cm.diffusion.karras_schedule")
def test_karras_sigmas_endpoints():
    """sigmas[0] == sigma_min and sigmas[-1] == sigma_max."""


@pytest.mark.skip(reason="implement after cm.diffusion.karras_schedule")
def test_karras_sigmas_monotonic_increasing():
    """All consecutive diffs > 0."""


@pytest.mark.skip(reason="implement after cm.diffusion.karras_schedule")
def test_n_schedule_endpoints():
    """n_schedule(0) == s0 and n_schedule(total_steps) == s1."""
