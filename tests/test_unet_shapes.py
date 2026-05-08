import pytest


@pytest.mark.skip(reason="implement after cm.models.unet")
def test_unet_forward_shape_cifar():
    """UNet(3,32,32) -> (3,32,32) for a batch of 4 with random sigmas."""


@pytest.mark.skip(reason="implement after cm.models.unet")
def test_unet_handles_scalar_sigma_broadcast():
    """Passing sigma as shape () or (B,) both work."""
