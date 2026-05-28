import argparse
import os
from pathlib import Path

import torchvision
from tqdm import tqdm


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


def prepare_imagenet64(data_dir):
    """ImageNet-64 prep — not implemented. Use HuggingFace Hub or your own pipeline."""
    print("[!] ImageNet64 download logic should be implemented here (e.g., via HuggingFace Hub).")


def prepare_lsun(data_dir, category: str = "church_outdoor"):
    """LSUN prep — not implemented. Use the official LMDB and extract to image folders."""
    print(f"[!] LSUN ({category}) download logic should be implemented here.")
    print("    LSUN requires massive disk space. Download LMDB and extract to standard image folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare datasets for training.")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "imagenet64", "lsun"])
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset == "cifar10":
        prepare_cifar10(args.data_dir)
    elif args.dataset == "imagenet64":
        prepare_imagenet64(args.data_dir)
    elif args.dataset == "lsun":
        prepare_lsun(args.data_dir)
