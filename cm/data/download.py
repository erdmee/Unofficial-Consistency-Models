import argparse
import multiprocessing
import os
import shutil
import subprocess
from pathlib import Path

import torchvision
from PIL import Image
from tqdm import tqdm


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _resize_one(args):
    """Worker: openai/consistency_models `center_crop_arr` preprocessing.

    Iterative BOX halve while shorter side >= 2*target → BICUBIC scale shorter
    side to target → center-crop to target×target. Idempotent (skips if dst exists).
    """
    src_path, dst_path, resolution = args
    if os.path.exists(dst_path):
        return True
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            while min(img.size) >= 2 * resolution:
                img = img.resize(
                    tuple(x // 2 for x in img.size),
                    resample=Image.Resampling.BOX,
                )
            scale = resolution / min(img.size)
            new_size = tuple(round(x * scale) for x in img.size)
            img = img.resize(new_size, resample=Image.Resampling.BICUBIC)
            w, h = img.size
            left = (w - resolution) // 2
            top = (h - resolution) // 2
            img = img.crop((left, top, left + resolution, top + resolution))
            img.save(dst_path, format="PNG")
        return True
    except Exception as e:
        print(f"[!] Failed {src_path}: {e}")
        return False


def prepare_cifar10(data_dir):
    """Download CIFAR-10 and write it out as data_dir/cifar10/{train,val}/<class>/<idx>.png."""
    base_path = Path(data_dir) / "cifar10"

    if base_path.exists():
        print(f"[*] CIFAR-10 already exists at {base_path}. Skipping download.")
        return

    print("[*] Downloading and extracting CIFAR-10...")

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    def _save_dataset(dataset, split_name):
        split_path = base_path / split_name
        for idx, (img, label) in enumerate(tqdm(dataset, desc=f"Saving {split_name} images")):
            class_name = dataset.classes[label]
            class_dir = split_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img.save(class_dir / f"{idx:05d}.png", format="PNG")

    _save_dataset(trainset, "train")
    _save_dataset(testset, "val")
    print(f"[*] CIFAR-10 preparation complete. Saved to {base_path}")


def download_imagenet_kaggle(raw_dir: str, auto_unzip: bool = True) -> str:
    """Download ILSVRC2012 (~168 GB) from Kaggle via the `kaggle` CLI.

    Prerequisites (must be done by user, cannot be automated):
      1. `pip install kaggle` (or have the `kaggle` CLI on PATH).
      2. ~/.kaggle/kaggle.json with API credentials, chmod 600.
         Get the token from https://www.kaggle.com/settings → Create New API Token.
      3. Accept the competition rules at
         https://www.kaggle.com/c/imagenet-object-localization-challenge/rules
         (the API will 403 until this is done in a browser).

    Returns the train-split path: {raw_dir}/ILSVRC/Data/CLS-LOC/train.
    Feed that path to `prepare_imagenet64(--source_dir ...)` for resizing.

    Disk budget reminder: ~168 GB zip + ~150 GB extracted ≈ 320 GB peak.
    Delete the zip / raw extract after resize to reclaim space.
    """
    if shutil.which("kaggle") is None:
        print("[!] `kaggle` CLI not found on PATH. Install with `pip install kaggle`.")
        return ""

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    train_dir = raw_path / "ILSVRC" / "Data" / "CLS-LOC" / "train"
    if train_dir.is_dir() and any(train_dir.iterdir()):
        print(f"[*] Train split already present at {train_dir}. Skipping download.")
        return str(train_dir)

    cmd = [
        "kaggle", "competitions", "download",
        "-c", "imagenet-object-localization-challenge",
        "-p", str(raw_path),
    ]
    if auto_unzip:
        cmd.append("--unzip")

    print(f"[*] Running: {' '.join(cmd)}")
    print("[*] This will download ~168 GB. Expect hours depending on network speed.")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[!] Kaggle CLI exited with code {result.returncode}.")
        print("[!] Common causes:")
        print("    - Missing/invalid credentials: place kaggle.json at ~/.kaggle/ (chmod 600).")
        print("    - Competition rules not accepted: open the competition page in a browser first.")
        print("    - Disk full: 168 GB zip + ~150 GB extract needs ~320 GB headroom.")
        return ""

    if not train_dir.is_dir():
        zip_path = raw_path / "imagenet-object-localization-challenge.zip"
        if zip_path.is_file():
            print("[!] Zip downloaded but not extracted. Re-run with auto_unzip=True or extract manually:")
            print(f"    unzip -q {zip_path} -d {raw_path}")
        else:
            print(f"[!] Expected train dir not found at {train_dir}.")
        return ""

    print(f"[*] Download + extract complete. Train split: {train_dir}")
    return str(train_dir)


def prepare_imagenet64(
    data_dir: str,
    source_dir: str | None = None,
    resolution: int = 64,
    num_workers: int = 8,
):
    """Preprocess raw ImageNet via openai/consistency_models' `center_crop_arr` pipeline.

    Per-image steps (matches `cm/image_datasets.py:center_crop_arr` in openai/consistency_models):
      1. Iterative BOX halving while shorter side >= 2*resolution (anti-aliased downsample)
      2. BICUBIC scale so shorter side == resolution
      3. Center-crop to resolution×resolution
      4. Save as PNG

    Caller must obtain the raw ILSVRC2012 dataset separately (e.g. from Kaggle) and
    pass its root via `source_dir`. Two source layouts are supported:
      - {source_dir}/<wnid>/<img>.JPEG          (standard ImageFolder)
      - {source_dir}/<wnid>_<idx>.JPEG          (flat with WNID prefix, EDM convention)

    Output: {data_dir}/imagenet{resolution}/train/<wnid>/<stem>.png.
    Existing output files are skipped, so re-runs only process missing images.
    """
    if source_dir is None:
        print("[!] --source_dir is required for imagenet64 preparation.")
        print("    Pass the directory containing raw ILSVRC2012 images.")
        return

    src_root = Path(source_dir)
    if not src_root.is_dir():
        print(f"[!] source_dir does not exist: {src_root}")
        return

    out_root = Path(data_dir) / f"imagenet{resolution}" / "train"
    out_root.mkdir(parents=True, exist_ok=True)

    src_root_resolved = src_root.resolve()
    tasks = []
    print(f"[*] Scanning {src_root} for source images...")
    for src_file in src_root.rglob("*"):
        if not src_file.is_file() or src_file.suffix.lower() not in _IMAGE_EXTS:
            continue
        if src_file.parent.resolve() != src_root_resolved:
            wnid = src_file.parent.name
        else:
            wnid = src_file.stem.split("_")[0]
        dst_dir = out_root / wnid
        dst_dir.mkdir(exist_ok=True)
        dst_file = dst_dir / (src_file.stem + ".png")
        tasks.append((str(src_file), str(dst_file), resolution))

    if not tasks:
        print(f"[!] No images found under {src_root}.")
        return

    print(f"[*] Resizing {len(tasks)} images → {out_root} with {num_workers} workers...")
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_resize_one, tasks, chunksize=64),
            total=len(tasks),
            desc="resize",
        ):
            pass

    print(f"[*] ImageNet-{resolution} preparation complete. Output: {out_root}")


def prepare_lsun(data_dir, category: str = "church_outdoor"):
    """LSUN prep — not implemented. Use the official LMDB and extract to image folders."""
    print(f"[!] LSUN ({category}) download logic should be implemented here.")
    print("    LSUN requires massive disk space. Download LMDB and extract to standard image folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare datasets for training.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "imagenet_raw", "imagenet64", "lsun"],
        help="cifar10: torchvision download. imagenet_raw: kaggle download + extract. "
             "imagenet64: resize already-extracted raw ImageNet to 64x64. lsun: stub.",
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="./raw_imagenet",
        help="Where to put the Kaggle zip + extracted ILSVRC (for --dataset imagenet_raw).",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Raw ImageNet train directory for resize (required for --dataset imagenet64). "
             "After --dataset imagenet_raw, this is {raw_dir}/ILSVRC/Data/CLS-LOC/train.",
    )
    parser.add_argument("--resolution", type=int, default=64, help="Output resolution for imagenet64.")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers for resize.")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset == "cifar10":
        prepare_cifar10(args.data_dir)
    elif args.dataset == "imagenet_raw":
        download_imagenet_kaggle(raw_dir=args.raw_dir)
    elif args.dataset == "imagenet64":
        prepare_imagenet64(
            data_dir=args.data_dir,
            source_dir=args.source_dir,
            resolution=args.resolution,
            num_workers=args.num_workers,
        )
    elif args.dataset == "lsun":
        prepare_lsun(args.data_dir)
