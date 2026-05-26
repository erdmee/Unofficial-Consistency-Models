import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from abc import abstractmethod

def zero_module(module):
    """
    Initializes the parameters (weights and biases) of a module to zero.
    This is often used for the final layers of residual blocks to ensure 
    they act as an identity mapping at the start of training.
    
    Args:
        module (nn.Module): The PyTorch module to initialize.
        
    Returns:
        nn.Module: The initialized module.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
):
    """
    Creates sinusoidal timestep embeddings.

    Args:
        timesteps (torch.Tensor): 1-D Tensor of N indices, one per batch element.
        embedding_dim (int): The dimension of the output embedding vector.
        max_period (int): Controls the minimum frequency of the embeddings.
        
    Returns:
        torch.Tensor: An embedding tensor of shape (N, embedding_dim).
    """
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

############################################################################

class TimestepBlock(nn.Module):
    """
    Abstract base class for modules that require both an input tensor and a 
    timestep embedding.
    """
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to child modules 
    if they support it (i.e., if they inherit from TimestepBlock).
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

############################################################################

class Downsample(nn.Module):
    """
    Downsample layer that reduces the spatial dimensions of the input.
    use_conv : True : 2d convoultion / False : average pooling
    """
    def __init__(
        self, 
        channels: int, 
        use_conv: bool = True, 
        out_channels: Optional[int] = None
    ):
        """
        Args:
            channels (int): Number of input channels.
            use_conv (bool): If True, uses a strided 2D convolution. 
                             If False, uses 2D average pooling.
            out_channels (Optional[int]): Number of output channels. Defaults to `channels`.
        """
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
                padding=1
            )
        else:
            assert self.channels == self.out_channels, "out_channels must be equal to channels when use_conv is False" 
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, channels, H, W).
            
        Returns:
            torch.Tensor: Downsampled tensor of shape (B, out_channels, H/2, W/2).
        """
        assert x.shape[1] == self.channels, f"expected {self.channels}, got {x.shape[1]}"
        return self.op(x)

class Upsample(nn.Module):
    """
    Upsample layer that increases the spatial dimensions of the input.
    use_conv : True : use convoluton after applying linear interpolation / False :  X
    """
    def __init__(
        self, 
        channels: int, 
        use_conv: bool = True, 
        out_channels: Optional[int] = None
    ):
        """
        Args:
            channels (int): Number of input channels.
            use_conv (bool): If True, applies a convolution after interpolation 
                             to correct artifacts.
            out_channels (Optional[int]): Number of output channels. Defaults to `channels`.
        """
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
                padding=1
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, channels, H, W).
            
        Returns:
            torch.Tensor: Upsampled tensor of shape (B, out_channels, H*2, W*2).
        """
        assert x.shape[1] == self.channels, f"expected {self.channels}, got {x.shape[1]}"
        
        # nearest interpolation : (H,W) -> (2H,2W)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        
        # Optional convolutional layer for pixel correction
        if self.use_conv:
            x = self.conv(x)
            
        return x

############################################################################

class ResBlock(TimestepBlock):
    """
    A residual block that incorporates timestep embeddings.
    """
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        out_channels: int = None,
        use_scale_shift_norm=False,
        up=False,   # upsampling mode
        down=False, # downsampling mode
    ):
        """
        Args:
            channels (int): Number of input channels.
            emb_channels (int): Dimension of the timestep embedding.
            dropout (float): Dropout probability.
            out_channels (Optional[int]): Number of output channels. Defaults to `channels`.
            use_scale_shift_norm (bool): If True, uses scale and shift in the normalization 
                                         layer following EDM's design. If False, uses standard GroupNorm.
            up (bool): If True, this block performs upsampling.
            down (bool): If True, this block performs downsampling.
        """
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.up = up
        self.down = down

        # Resolution adjustment layer (Upsample, Downsample, or Identity)
        if self.up:
            self.resizer = Upsample(channels=channels, use_conv=False)
        elif self.down:
            self.resizer = Downsample(channels=channels, use_conv=False)
        else:
            self.resizer = nn.Identity()

        # 1. First convolutional block(GroupNorm -> SiLU -> Conv)
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )

        # 2. Time embedding projection
        emb_out_channels = self.out_channels * 2 if use_scale_shift_norm else self.out_channels
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, emb_out_channels)
        )

        # 3. Output convolutional block (with dropout and zero-initialized weights)
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1))
        )

        # 4. Skip connection matching
        if self.channels == self.out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(self.channels, self.out_channels, kernel_size=1)


    def forward(self, x, emb):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, channels, H, W).
            emb (torch.Tensor): Timestep embedding tensor of shape (B, emb_channels).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        # 1. Calculate x_skip (including resolution adjustment)
        x_skip = self.resizer(x)

        # 2. h = GroupNorm -> SiLU -> Conv
        if self.up or self.down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.resizer(h)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # 3. scale-shift normalization 
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
    """
    A multi-head attention block utilizing PyTorch 2.0's highly optimized 
    scaled dot-product attention (SDPA).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
    ):
        """
        Args:
            channels (int): Number of input channels.
            num_heads (int): Number of attention heads. Used if `num_head_channels` is -1.
            num_head_channels (int): Number of channels per head. If specified, overrides `num_heads`.
        """
        super().__init__()
        self.channels = channels
        
        # Determine the number of heads based on provided arguments
        if num_head_channels != -1:
            self.num_heads = channels // num_head_channels
        else:
            self.num_heads = num_heads

        # 1. GroupNorm
        self.norm = nn.GroupNorm(32, channels)

        # 2. 1x1 Conv to produce Q, K, V in one go
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

        # 3. 1x1 Conv to project the output (with zero initialization for stable training)
        self.proj_out = zero_module(nn.Conv2d(channels, channels, kernel_size=1))


    def forward(self, x, emb):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            emb (Optional[torch.Tensor]): Not used directly in this block, but included 
                                          to comply with the TimestepBlock interface.
                                          
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        
        # 1. GroupNorm and QKV extraction / shape: (B, 3C, H, W)
        qkv = self.qkv(self.norm(x))

        # 2. QKV partitoning and reshape for multi-head attention
        # (B, 3, num_heads, head_dim, Seq_len)
        qkv = qkv.view(B, 3, self.num_heads, C // self.num_heads, H * W)
        
        # Permute to: (3, B, num_heads, seq_len, head_dim)
        qkv = qkv.permute(1, 0, 2, 4, 3)      
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 3. Scaled Dot-Product Attention (SDPA) 
        # output shape: (B, num_heads, Seq_len, head_dim)
        attn_out = F.scaled_dot_product_attention(q, k, v)

        # 4. Reshape back to (B, C, H, W)
        # (B, num_heads, seq_len, head_dim) -> (B, num_heads, head_dim, seq_len)
        attn_out = attn_out.permute(0, 1, 3, 2).contiguous()
        attn_out = attn_out.view(B, C, H, W)

        # 5. Output projection and residual connection
        return x + self.proj_out(attn_out)
