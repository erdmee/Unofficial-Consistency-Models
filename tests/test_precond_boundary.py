import torch

from cm.models.precond import ConsistencyPrecond
from cm.models.unet import UNetModel


def _build(epsilon: float = 0.002) -> ConsistencyPrecond:
    """A tiny U-Net so the test runs in a second or two on CPU."""
    unet = UNetModel(
        in_channels=3,
        model_channels=32,
        out_channels=3,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(),
    )
    return ConsistencyPrecond(unet, epsilon=epsilon).eval()


def test_consistency_model_boundary_condition():
    """f_theta(x, sigma=epsilon) == x.

    At t = epsilon: c_skip = sigma_data² / sigma_data² = 1, c_out = 0.
    So out = 1 * x + 0 * model_out = x, independent of the network weights.
    """
    torch.manual_seed(0)
    model = _build(epsilon=0.002)
    x = torch.randn(2, 3, 8, 8)
    t = torch.full((2,), 0.002)
    with torch.no_grad():
        out = model(x, t)
    assert torch.allclose(out, x, atol=1e-6, rtol=0.0)


def test_consistency_model_output_shape():
    """Output shape == input shape across a batch of mixed sigmas."""
    torch.manual_seed(0)
    model = _build()
    x = torch.randn(3, 3, 8, 8)
    t = torch.tensor([0.5, 5.0, 50.0])
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
