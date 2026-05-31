import argparse
import math

import torch
import torch.nn as nn
from torchvision.utils import save_image

from cm.sampling.loader import load_consistency_model, resolve_labels

# Paper-recommended CIFAR-10 ts schedules, keyed by NFE (= 1 initial denoise + len(ts) steps).
NFE_TS_PRESETS = {
    2: [0.821],
    4: [24.4, 5.84, 0.9],
}


@torch.no_grad()
def generate_multistep(
    model: nn.Module,
    batch_size: int,
    image_size: int,
    device: torch.device,
    ts: list[float],
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-step consistency sampling. `ts` is the descending list of intermediate noise
    levels, each strictly in (sigma_min, sigma_max). Returns images in [-1, 1]."""
    model.eval()
    shape = (batch_size, 3, image_size, image_size)

    x = torch.randn(*shape, device=device) * sigma_max

    t_init = torch.full((batch_size,), sigma_max, device=device)
    x = model(x, t_init, y)

    for tau in ts:
        if not (sigma_min < tau < sigma_max):
            raise ValueError(
                f"tau={tau} must lie strictly between sigma_min={sigma_min} "
                f"and sigma_max={sigma_max}"
            )

        z = torch.randn_like(x)
        x = x + z * math.sqrt(tau ** 2 - sigma_min ** 2)

        t = torch.full((batch_size,), tau, device=device)
        x = model(x, t, y)

    return torch.clamp(x, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Multi-Step Sampling for Consistency Models")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained checkpoint (.pt)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Training config YAML used to reconstruct the architecture when the checkpoint has no "
             "embedded config (e.g. configs/cifar10_cd.yaml). Ignored if the checkpoint already has one.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Number of images to generate")
    parser.add_argument("--out_path", type=str, default="sample_multistep.png", help="Output image path")
    parser.add_argument(
        "--nfe",
        type=int,
        default=2,
        choices=sorted(NFE_TS_PRESETS),
        help="Number of function evaluations. Selects a paper-recommended CIFAR-10 ts schedule "
             "(2 -> [0.821], 4 -> [24.4, 5.84, 0.9]). Ignored if --ts is given.",
    )
    parser.add_argument(
        "--ts",
        type=str,
        default=None,
        help="Explicit descending noise levels, space- or comma-separated (e.g. '24.4 5.84 0.9'). "
             "Overrides --nfe. Each value must lie strictly in (sigma_min, sigma_max).",
    )
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
    schedule = config.get("schedule", {})
    sigma_min = schedule.get("sigma_min", 0.002)
    sigma_max = schedule.get("sigma_max", 80.0)
    num_classes = config["model"].get("num_classes")
    print(
        f"[*] Loaded sampling EMA model from step {step} "
        f"(image_size={image_size}, model_channels={config['model']['model_channels']}, "
        f"num_classes={num_classes})."
    )

    if args.ts is not None:
        ts = [float(v) for v in args.ts.replace(",", " ").split()]
        if not ts:
            raise ValueError("--ts was given but parsed to an empty list.")
    else:
        ts = NFE_TS_PRESETS[args.nfe]
        print(
            f"[*] Using preset ts for NFE={args.nfe}: {ts}. "
            f"NOTE: presets are tuned for CIFAR-10; pass --ts explicitly for other datasets."
        )
    print(f"[*] Multi-step sampling with ts={ts} (NFE={1 + len(ts)}).")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"[*] Using seed {args.seed}.")

    y = resolve_labels(num_classes, args.batch_size, args.class_id, device)

    print("[*] Performing multi-step generation...")
    generated_images = generate_multistep(
        model=model,
        batch_size=args.batch_size,
        image_size=image_size,
        device=device,
        ts=ts,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        y=y,
    )

    # De-normalize from [-1, 1] to [0, 1] for saving
    images_denorm = (generated_images + 1.0) / 2.0
    save_image(images_denorm, args.out_path, nrow=8)
    print(f"[*] Saved {args.batch_size} images to '{args.out_path}'.")


if __name__ == "__main__":
    main()
