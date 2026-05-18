import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Generator, Tuple, Dict
import torch

# Import from our new decoupled modules
from transforms import build_transforms
from dataset import ImageDataset

def _list_image_files_recursively(data_dir: str):
    """Recursively lists all standard image files within a directory."""
    results = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                results.append(os.path.join(root, file))
    return sorted(results)

def create_data_loader(
    data_dir: str,
    batch_size: int,
    image_size: int,
    class_cond: bool = False,
    deterministic: bool = False,
    random_crop: bool = False,
    random_flip: bool = True,
    num_workers: int = 4,
):
    """
    Orchestrates file discovery, sharding, transform building, 
    and returns an infinite step-based data generator.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # 1. Gather all files and classes
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    
    if class_cond:
        class_names = [os.path.basename(os.path.dirname(path)) for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    # 2. Handle Distributed Data Parallel (DDP) sharding
    shard = 0
    num_shards = 1
    if dist.is_available() and dist.is_initialized():
        shard = dist.get_rank()
        num_shards = dist.get_world_size()

    local_files = all_files[shard:][::num_shards]
    local_classes = classes[shard:][::num_shards] if classes is not None else None

    # 3. Build Transforms
    transform_pipeline = build_transforms(
        resolution=image_size,
        random_crop=random_crop,
        random_flip=random_flip
    )

    # 4. Initialize Pure Dataset
    dataset = ImageDataset(
        image_paths=local_files,
        transform=transform_pipeline,
        classes=local_classes
    )

    # 5. Initialize PyTorch DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=not deterministic, 
        num_workers=num_workers, 
        drop_last=True,
        pin_memory=True
    )

    # 6. Yield as an infinite generator for step-based training
    while True:
        yield from loader