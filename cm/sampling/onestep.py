import argparse

import torch
from torchvision.utils import save_image

from cm.models.unet import UNetModel
from cm.models.precond import ConsistencyPrecond
from cm.utils.checkpoint import load_checkpoint


@torch.no_grad()
def generate_one_step(
    model: torch.nn.Module,
    batch_size: int,
    image_size: int,
    device: torch.device,
    sigma_max: float = 80.0,
):
    """1-step generation from a trained Consistency Model. Returns images in [-1, 1]."""
    model.eval()

    shape = (batch_size, 3, image_size, image_size)
    x_T = torch.randn(*shape, device=device) * sigma_max
    t_tensor = torch.full((batch_size,), sigma_max, device=device)
    x_0 = model(x_T, t_tensor)

    return torch.clamp(x_0, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="1-Step Sampling for Consistency Models")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of images to generate")
    parser.add_argument("--image_size", type=int, default=32, help="Image resolution")
    parser.add_argument("--out_path", type=str, default="sample_1step.png", help="Output image path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Generating on device: {device}")

    unet = UNetModel(in_channels=3, model_channels=128, out_channels=3)
    target_model = ConsistencyPrecond(unet).to(device)

    print(f"[*] Loading checkpoint from {args.ckpt}...")
    step, _ = load_checkpoint(
        load_path=args.ckpt,
        ema_model=target_model,
        model=None,
        optimizer=None,
        device=str(device),
    )
    print(f"[*] Successfully loaded EMA model from step {step}.")

    print("[*] Performing 1-step generation...")
    generated_images = generate_one_step(
        model=target_model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        sigma_max=80.0,
    )

    # De-normalize from [-1, 1] to [0, 1] for saving
    images_denorm = (generated_images + 1.0) / 2.0
    save_image(images_denorm, args.out_path, nrow=8)
    print(f"[*] Saved {args.batch_size} images to '{args.out_path}'.")


if __name__ == "__main__":
    main()
