from typing import Callable, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Loads images from a path list and applies a single transform pipeline."""
    def __init__(
        self,
        image_paths: List[str],
        transform: Callable,
        classes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.local_images = image_paths
        self.local_classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        with open(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        pil_image = pil_image.convert("RGB")
        tensor = self.transform(pil_image)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = torch.tensor(self.local_classes[idx], dtype=torch.long)

        return tensor, out_dict
