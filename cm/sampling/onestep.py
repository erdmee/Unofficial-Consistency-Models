import argparse

import torch
from torchvision.utils import save_image

from cm.sampling.loader import load_consistency_model, resolve_labels


@torch.no_grad()
def generate_one_step(
    model: torch.nn.Module,
    batch_size: int,
    image_size: int,
    device: torch.device,
    sigma_max: float = 80.0,
    y: torch.Tensor | None = None,
):
    """1-step generation from a Consistency Model. Returns images in [-1, 1].
    `y` is the class-label tensor for class-conditional models (None otherwise)."""
    model.eval()

    shape = (batch_size, 3, image_size, image_size)
    x_T = torch.randn(*shape, device=device) * sigma_max
    t_tensor = torch.full((batch_size,), sigma_max, device=device)
    x_0 = model(x_T, t_tensor, y)

    return torch.clamp(x_0, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="1-Step Sampling for Consistency Models")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained checkpoint (.pt)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training config YAML used to reconstruct the architecture when the checkpoint has no "
             "embedded config (e.g. configs/cifar10_cd.yaml). Ignored if the checkpoint already has one.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Number of images to generate")
    parser.add_argument("--out_path", type=str, default="sample.png", help="Output image path")
    parser.add_argument(
        "--class_id",
        type=int,
        default=None,
        help="Class label for class-conditional models (e.g. ImageNet-64), applied to all samples. "
             "Omit to sample random classes. Must be omitted for unconditional models (e.g. CIFAR-10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible noise (and random class labels). Omit for fresh randomness.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Generating on device: {device}")

    print(f"[*] Loading checkpoint from {args.ckpt}...")
    model, config, step = load_consistency_model(args.ckpt, device, args.config)

    image_size = config["data"]["image_size"]
    sigma_max = config.get("schedule", {}).get("sigma_max", 80.0)
    num_classes = config["model"].get("num_classes")
    print(
        f"[*] Loaded sampling EMA model from step {step} "
        f"(image_size={image_size}, model_channels={config['model']['model_channels']}, "
        f"num_classes={num_classes})."
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"[*] Using seed {args.seed}.")

    y = resolve_labels(num_classes, args.batch_size, args.class_id, device)

    print("[*] Performing 1-step generation...")
    generated_images = generate_one_step(
        model=model,
        batch_size=args.batch_size,
        image_size=image_size,
        device=device,
        sigma_max=sigma_max,
        y=y,
    )

    # De-normalize from [-1, 1] to [0, 1] for saving
    images_denorm = (generated_images + 1.0) / 2.0
    save_image(images_denorm, args.out_path, nrow=8)
    print(f"[*] Saved {args.batch_size} images to '{args.out_path}'.")


if __name__ == "__main__":
    main()
