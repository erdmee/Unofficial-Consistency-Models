import torch

from cm.diffusion.karras_schedule import karras_sigmas, n_schedule


def test_karras_sigmas_endpoints():
    """sigmas[0] == sigma_min and sigmas[-1] == sigma_max (ascending convention)."""
    sigma_min, sigma_max = 0.002, 80.0
    s = karras_sigmas(N=40, sigma_min=sigma_min, sigma_max=sigma_max)
    assert s.shape == (40,)
    assert torch.isclose(s[0], torch.tensor(sigma_min), atol=1e-6)
    assert torch.isclose(s[-1], torch.tensor(sigma_max), atol=1e-4)


def test_karras_sigmas_monotonic_increasing():
    """All consecutive diffs > 0."""
    s = karras_sigmas(N=40, sigma_min=0.002, sigma_max=80.0)
    diffs = s[1:] - s[:-1]
    assert (diffs > 0).all()


def test_n_schedule_endpoints():
    """N(0) == s0 and N(K) == s1 + 1 (CT paper Eq. 11, "+1 boundary points")."""
    s0, s1, total = 2, 150, 1000
    assert n_schedule(0, total, s0, s1) == s0
    assert n_schedule(total, total, s0, s1) == s1 + 1


def test_n_schedule_monotonic_non_decreasing():
    """N(k) never decreases — buckets only grow as training progresses."""
    s0, s1, total = 2, 150, 1000
    prev = n_schedule(0, total, s0, s1)
    for k in range(0, total + 1, 25):
        cur = n_schedule(k, total, s0, s1)
        assert cur >= prev, f"k={k}: N decreased from {prev} to {cur}"
        prev = cur
