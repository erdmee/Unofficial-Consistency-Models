"""
InceptionV3 feature extractor for FID.

We load the *FID Inception* weights — i.e. the TF inception checkpoint that
Heusel et al. 2017 used and that every diffusion paper since (DDPM, ADM, EDM,
CM, ...) reports FID against. These weights are hosted by `pytorch-fid`
(MIT-licensed) at:

    https://github.com/mseitzer/pytorch-fid

Architecture-wise, the TF FID Inception differs from torchvision's stock
Inception V3 in a few avg-pool quirks (count_include_pad=False; one max_pool
swap in the last block). We start from torchvision's Inception V3 and
patch those specific blocks so the saved state_dict loads cleanly.

Input convention: this repo keeps images in [-1, 1] (EDM/CM). The FID
Inception's first conv was trained expecting [-1, 1] as well, so no
ImageNet-style mean/std normalization is applied here — we only resize
to 299x299.
"""

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3 as _tv_inception_v3
from torchvision.models.inception import InceptionA, InceptionC, InceptionE


FID_WEIGHTS_URL = (
    "https://github.com/mseitzer/pytorch-fid/releases/download/"
    "fid_weights/pt_inception-2015-12-05-6726825d.pth"
)


def _avg_pool_tf(x: torch.Tensor) -> torch.Tensor:
    """3x3 avg pool with count_include_pad=False (TF default)."""
    return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)


# ---------------------------------------------------------------------------
# Block variants — same parameters as torchvision's blocks, but with the TF
# pooling convention so the FID-Inception state_dict loads exactly.
# ---------------------------------------------------------------------------

class _FIDInceptionA(InceptionA):
    def _forward(self, x):
        b1 = self.branch1x1(x)
        b5 = self.branch5x5_2(self.branch5x5_1(x))
        b3 = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        bp = self.branch_pool(_avg_pool_tf(x))
        return [b1, b5, b3, bp]


class _FIDInceptionC(InceptionC):
    def _forward(self, x):
        b1 = self.branch1x1(x)
        b7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        b7d = self.branch7x7dbl_1(x)
        b7d = self.branch7x7dbl_2(b7d)
        b7d = self.branch7x7dbl_3(b7d)
        b7d = self.branch7x7dbl_4(b7d)
        b7d = self.branch7x7dbl_5(b7d)
        bp = self.branch_pool(_avg_pool_tf(x))
        return [b1, b7, b7d, bp]


class _FIDInceptionE_1(InceptionE):
    """Penultimate block — TF avg pool."""
    def _forward(self, x):
        b1 = self.branch1x1(x)
        b3 = self.branch3x3_1(x)
        b3 = torch.cat([self.branch3x3_2a(b3), self.branch3x3_2b(b3)], 1)
        b3d = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        b3d = torch.cat([self.branch3x3dbl_3a(b3d), self.branch3x3dbl_3b(b3d)], 1)
        bp = self.branch_pool(_avg_pool_tf(x))
        return [b1, b3, b3d, bp]


class _FIDInceptionE_2(InceptionE):
    """Final block — TF graph uses MAX pool here, not avg."""
    def _forward(self, x):
        b1 = self.branch1x1(x)
        b3 = self.branch3x3_1(x)
        b3 = torch.cat([self.branch3x3_2a(b3), self.branch3x3_2b(b3)], 1)
        b3d = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        b3d = torch.cat([self.branch3x3dbl_3a(b3d), self.branch3x3dbl_3b(b3d)], 1)
        bp = self.branch_pool(F.max_pool2d(x, kernel_size=3, stride=1, padding=1))
        return [b1, b3, b3d, bp]


def _build_fid_inception() -> nn.Module:
    # FID checkpoint has 1008-way classifier head and no aux logits.
    net = _tv_inception_v3(
        num_classes=1008,
        aux_logits=False,
        weights=None,
        init_weights=False,
    )
    net.Mixed_5b = _FIDInceptionA(192, pool_features=32)
    net.Mixed_5c = _FIDInceptionA(256, pool_features=64)
    net.Mixed_5d = _FIDInceptionA(288, pool_features=64)
    net.Mixed_6b = _FIDInceptionC(768, channels_7x7=128)
    net.Mixed_6c = _FIDInceptionC(768, channels_7x7=160)
    net.Mixed_6d = _FIDInceptionC(768, channels_7x7=160)
    net.Mixed_6e = _FIDInceptionC(768, channels_7x7=192)
    net.Mixed_7b = _FIDInceptionE_1(1280)
    net.Mixed_7c = _FIDInceptionE_2(2048)

    state_dict = torch.hub.load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    net.load_state_dict(state_dict)
    return net


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """
    Reference FID Inception (2048-d pool3 features).

    Args:
        device: CUDA device to place the extractor on. Defaults to current cuda device.

    Forward:
        images: (N, 3, H, W) in [-1, 1].
        returns: (N, 2048).
    """

    POOL_DIM: int = 2048
    IMAGE_SIZE: int = 299

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        net = _build_fid_inception()

        # We only need everything up to the final 8x8 -> 1x1 avg pool, so
        # discard the classifier. The pool itself is applied in forward.
        net.fc = nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net

        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.to(device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError(f"expected (N,3,H,W), got {tuple(images.shape)}")
        x = images
        if x.shape[-1] != self.IMAGE_SIZE or x.shape[-2] != self.IMAGE_SIZE:
            x = F.interpolate(
                x,
                size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        feats = self.net(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        return feats

    @torch.no_grad()
    def extract(self, images_iter: Iterable[torch.Tensor]) -> torch.Tensor:
        """Run the extractor over an iterable of image batches; return (N, 2048) on the extractor's device."""
        device = next(self.parameters()).device
        chunks = []
        for batch in images_iter:
            batch = batch.to(device, non_blocking=True)
            chunks.append(self(batch))
        return torch.cat(chunks, dim=0)
