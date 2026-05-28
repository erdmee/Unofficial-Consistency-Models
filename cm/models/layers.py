import math
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    """Zero-initialize parameters so the module starts as an identity contribution."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
):
    """Sinusoidal timestep embeddings (cos | sin)."""
    assert len(timesteps.shape) == 1, "timesteps must be a 1-D tensor"

    half_dim = embedding_dim // 2
    emb = math.log(max_period) / half_dim
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential that forwards the timestep embedding to child TimestepBlocks."""
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: int | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if self.use_conv:
            self.op = nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            assert self.channels == self.out_channels, "out_channels must equal channels when use_conv=False"
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        assert x.shape[1] == self.channels, f"expected {self.channels}, got {x.shape[1]}"
        return self.op(x)


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x):
        assert x.shape[1] == self.channels, f"expected {self.channels}, got {x.shape[1]}"
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        out_channels: int | None = None,
        use_scale_shift_norm=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.up = up
        self.down = down

        if self.up:
            self.resizer = Upsample(channels=channels, use_conv=False)
        elif self.down:
            self.resizer = Downsample(channels=channels, use_conv=False)
        else:
            self.resizer = nn.Identity()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )

        emb_out_channels = self.out_channels * 2 if use_scale_shift_norm else self.out_channels
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, emb_out_channels),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)),
        )

        if self.channels == self.out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)

    def forward(self, x, emb):
        x_skip = self.resizer(x)

        if self.up or self.down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.resizer(h)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x_skip) + h


class AttentionBlock(TimestepBlock):
    """Multi-head self-attention via PyTorch SDPA. `emb` is ignored (TimestepBlock interface)."""
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels != -1:
            self.num_heads = channels // num_head_channels
        else:
            self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = zero_module(nn.Conv2d(channels, channels, kernel_size=1))

    def forward(self, x, emb):
        B, C, H, W = x.shape

        qkv = self.qkv(self.norm(x))
        qkv = qkv.view(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.permute(0, 1, 3, 2).contiguous()
        attn_out = attn_out.view(B, C, H, W)

        return x + self.proj_out(attn_out)
