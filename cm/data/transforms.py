from torchvision import transforms

def build_transforms(
    resolution: int, 
    random_crop: bool = False, 
    random_flip: bool = True
):
    """
    Builds the torchvision transform pipeline.
    """
    transform_list = []

    if random_crop:
        transform_list.append(
            transforms.RandomResizedCrop(
                size=resolution,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            )
        )
    else:
        transform_list.extend([
            transforms.Resize(
                size=resolution, 
                interpolation=transforms.InterpolationMode.BICUBIC, 
                antialias=True
            ),
            transforms.CenterCrop(resolution)
        ])

    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transforms.Compose(transform_list)