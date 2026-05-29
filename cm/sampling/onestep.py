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
    y: torch.Tensor | None = None,
):
    """1-step generation from a trained Consistency Model. Returns images in [-1, 1].

    `y` is the per-sample class label tensor for class-conditional models (None for unconditional).
    """
    model.eval()

    shape = (batch_size, 3, image_size, image_size)
    x_T = torch.randn(*shape, device=device) * sigma_max
    t_tensor = torch.full((batch_size,), sigma_max, device=device)
    x_0 = model(x_T, t_tensor, y)

    return torch.clamp(x_0, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="1-Step Sampling for Consistency Models")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of images to generate")
    parser.add_argument("--image_size", type=int, default=32, help="Image resolution")
    parser.add_argument("--out_path", type=str, default="sample_1step.png", help="Output image path")
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="Class label for class-conditional models. Applied to all samples in the batch. "
             "Omit for unconditional models.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes the model was trained with (required for class-conditional checkpoints).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Generating on device: {device}")

    unet_kwargs = dict(in_channels=3, model_channels=128, out_channels=3, num_classes=args.num_classes)
    target_model = ConsistencyPrecond(UNetModel(**unet_kwargs)).to(device)
    sampling_ema_model = ConsistencyPrecond(UNetModel(**unet_kwargs)).to(device)

    print(f"[*] Loading checkpoint from {args.ckpt}...")
    step, _, _ = load_checkpoint(
        load_path=args.ckpt,
        ema_model=target_model,
        model=None,
        optimizer=None,
        sampling_ema_model=sampling_ema_model,
        device=str(device),
    )
    print(f"[*] Successfully loaded sampling EMA model from step {step}.")

    y = None
    if args.class_id is not None:
        if args.num_classes is None:
            raise ValueError("--class_id requires --num_classes for class-conditional sampling.")
        y = torch.full((args.batch_size,), args.class_id, dtype=torch.long, device=device)
        print(f"[*] Class-conditional sampling with class_id={args.class_id}")

    print("[*] Performing 1-step generation...")
    generated_images = generate_one_step(
        model=sampling_ema_model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        sigma_max=80.0,
        y=y,
    )

    # De-normalize from [-1, 1] to [0, 1] for saving
    images_denorm = (generated_images + 1.0) / 2.0
    save_image(images_denorm, args.out_path, nrow=8)
    print(f"[*] Saved {args.batch_size} images to '{args.out_path}'.")


if __name__ == "__main__":
    main()
