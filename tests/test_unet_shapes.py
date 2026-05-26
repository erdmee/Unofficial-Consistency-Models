import pytest
import torch

from cm.models.unet import UNetModel


def _build(**kwargs) -> UNetModel:
    """Tiny U-Net sized for fast CPU tests (CIFAR-shaped but reduced channels)."""
    return UNetModel(
        in_channels=3,
        model_channels=32,
        out_channels=3,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(),
        **kwargs,
    )


def test_unet_forward_shape_cifar():
    """UNet(3,32,32) -> (3,32,32) for a batch of 4 with random sigmas."""
    torch.manual_seed(0)
    model = _build().eval()
    x = torch.randn(4, 3, 32, 32)
    t = torch.rand(4) * 80.0
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_unet_handles_scalar_sigma_broadcast():
    """A constant sigma replicated to shape (B,) works; a 0-D scalar is rejected."""
    torch.manual_seed(0)
    model = _build().eval()
    x = torch.randn(2, 3, 8, 8)

    # (B,) constant — the supported broadcast pattern.
    t_full = torch.full((2,), 5.0)
    with torch.no_grad():
        out = model(x, t_full)
    assert out.shape == x.shape

    # 0-D scalar must fail — get_timestep_embedding asserts 1-D input.
    with pytest.raises(AssertionError):
        model(x, torch.tensor(5.0))


def test_unet_class_conditioning_shape():
    """num_classes set + label tensor (B,) -> output shape preserved."""
    torch.manual_seed(0)
    model = _build(num_classes=10).eval()
    x = torch.randn(2, 3, 8, 8)
    t = torch.full((2,), 5.0)
    y = torch.tensor([0, 7])
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == x.shape
