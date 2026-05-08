import pytest


@pytest.mark.skip(reason="implement after cm.sampling.onestep")
def test_onestep_sample_shape():
    """onestep_sample(model, n=4, shape=(3,32,32)) -> (4,3,32,32)."""


@pytest.mark.skip(reason="implement after cm.sampling.multistep")
def test_multistep_sample_shape_and_range():
    """multistep_sample output shape matches and values are finite."""
