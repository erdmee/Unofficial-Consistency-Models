import torch

from cm.models.precond import ConsistencyPrecond
from cm.models.unet import UNetModel
from cm.sampling.multistep import generate_multistep
from cm.sampling.onestep import generate_one_step


def _build_model(device: torch.device) -> ConsistencyPrecond:
    unet = UNetModel(
        in_channels=3,
        model_channels=32,
        out_channels=3,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(),
    )
    return ConsistencyPrecond(unet).to(device).eval()


def test_onestep_sample_shape():
    """onestep -> (B, 3, H, W), values clamped to [-1, 1]."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _build_model(device)
    out = generate_one_step(model, batch_size=4, image_size=8, device=device)
    assert out.shape == (4, 3, 8, 8)
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0
    assert torch.isfinite(out).all()


def test_multistep_sample_shape_and_range():
    """multistep output shape matches and values are finite, clamped to [-1, 1]."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _build_model(device)
    # Paper-recommended CIFAR-10 NFE=4 schedule
    ts = [24.4, 5.84, 0.9]
    out = generate_multistep(
        model,
        batch_size=2,
        image_size=8,
        device=device,
        ts=ts,
    )
    assert out.shape == (2, 3, 8, 8)
    assert torch.isfinite(out).all()
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0
