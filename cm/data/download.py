import os
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torchvision

def prepare_cifar10(data_dir):
    """
    Downloads CIFAR-10 and extracts it into standard ImageFolder structure:
    data_dir/cifar10/train/class_name/0001.png
    """
    dataset_name = "cifar10"
    base_path = Path(data_dir) / dataset_name
    
    if base_path.exists():
        print(f"[*] CIFAR-10 already exists at {base_path}. Skipping download.")
        return

    print("[*] Downloading and extracting CIFAR-10...")
    
    # Use torchvision to download the raw dataset (pickled format)
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Helper function to save images to disk
    def _save_dataset(dataset, split_name):
        split_path = base_path / split_name
        for idx, (img, label) in enumerate(tqdm(dataset, desc=f"Saving {split_name} images")):
            class_name = dataset.classes[label]
            class_dir = split_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Save PIL Image as PNG
            img_path = class_dir / f"{idx:05d}.png"
            img.save(img_path, format="PNG")

    _save_dataset(trainset, "train")
    _save_dataset(testset, "val")
    print(f"[*] CIFAR-10 preparation complete. Saved to {base_path}")


def prepare_imagenet64(data_dir):
    """
    Placeholder for ImageNet64x64.
    Recommendation: Use huggingface_hub to download.
    """
    # Example using huggingface_hub (Requires pip install huggingface_hub)
    # from huggingface_hub import snapshot_download
    # print("[*] Downloading ImageNet-64 from Hugging Face...")
    # snapshot_download(repo_id="valhalla/imagenet-64-imagefolder", local_dir=os.path.join(data_dir, "imagenet64"))
    print("[!] ImageNet64 download logic should be implemented here (e.g., via HuggingFace Hub).")


def prepare_lsun(data_dir, category: str = "church_outdoor"):
    """
    Placeholder for LSUN.
    Recommendation: Use the official LSUN download script to get LMDB, then extract to images.
    """
    print(f"[!] LSUN ({category}) download logic should be implemented here.")
    print("    LSUN requires massive disk space. Download LMDB and extract to standard image folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare datasets for training.")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "imagenet64", "lsun"], 
                        help="The dataset to download and prepare.")
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Base directory to store datasets.")
    
    args = parser.parse_args()

    # Ensure base directory exists
    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset == "cifar10":
        prepare_cifar10(args.data_dir)
    elif args.dataset == "imagenet64":
        prepare_imagenet64(args.data_dir)
    elif args.dataset == "lsun":
        prepare_lsun(args.data_dir)