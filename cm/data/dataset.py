import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, List, Optional, Tuple, Dict

class ImageDataset(Dataset):
    """
    A pure Dataset class. It only handles loading the image from disk
    and applying the provided transform function.
    """
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

        # Apply the injected transform pipeline
        tensor = self.transform(pil_image)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = torch.tensor(self.local_classes[idx], dtype=torch.long)
            
        return tensor, out_dict