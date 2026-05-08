import pytest


@pytest.mark.skip(reason="implement after cm.models.precond")
def test_consistency_model_boundary_condition():
    """f_theta(x, sigma=sigma_min) must equal x exactly (the defining boundary
    condition of consistency models — c_skip(sigma_min)=1, c_out(sigma_min)=0)."""


@pytest.mark.skip(reason="implement after cm.models.precond")
def test_consistency_model_output_shape():
    """Output shape == input shape across a batch of mixed sigmas."""
