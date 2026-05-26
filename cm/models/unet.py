import torch
import torch.nn as nn

from cm.models.layers import (
    zero_module,
    get_timestep_embedding,
    TimestepEmbedSequential,
    Downsample,
    Upsample,
    ResBlock,
    AttentionBlock,
)

class UNetModel(nn.Module):
    """
    2D U-Net architecture with support for class conditioning.
    """
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        model_channels: int = 192,
        out_channels: int = 3,
        num_res_blocks: int = 3,
        attention_resolutions: tuple = (32, 16, 8), # Downsample factors to apply attention
        dropout: float = 0.1,
        channel_mult: tuple = (1, 2, 3, 4),
        num_heads: int = 4,
        num_head_channels: int = -1,
        num_classes: int = 1000, # Added parameter for class conditioning
        use_scale_shift_norm: bool = True, # following EDM's use of scale-shift in ResBlocks
        resblock_updown: bool = True, # Whether to use ResBlocks in up/downsample layers (EDM uses True)
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown

        # 1. Initialize Time Embedding layer
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 2. Initialize Class Embedding layer (if num_classes is specified)
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # 3. Input Convolution
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
            )
        ])
        
        # Keep track of channel dimensions for U-Net skip connections
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1  # Track current downsample factor

        # 4. Downsampling Blocks (Encoder)
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                
                # Apply attention if the current resolution matches specified factors
                if (self.image_size // ds) in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            
            # Apply Downsample if it's not the last level
            if level != len(channel_mult) - 1:
                # downsample with ResBlock if resblock_updown is True, otherwise use simple Downsample
                if self.resblock_updown:
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            ResBlock(
                                channels=ch, 
                                emb_channels=time_embed_dim, 
                                dropout=dropout, 
                                out_channels=ch, 
                                use_scale_shift_norm=self.use_scale_shift_norm, 
                                down=True # This will perform downsampling inside the ResBlock
                            )
                        )
                    )
                else:
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            Downsample(channels=ch, use_conv=True, out_channels=ch)
                        )
                    )
                input_block_chans.append(ch)
                ds *= 2

        # 5. Middle Blocks (Bottleneck)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=self.use_scale_shift_norm),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=self.use_scale_shift_norm),
        )

        # 6. Upsampling Blocks (Decoder)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # Pop the channel dimension coming from the encoder skip connection
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                
                if (self.image_size // ds) in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                
                # Apply Upsample if it's the last block of the level (and not the top level)
                if level and i == num_res_blocks:
                    # upsample with ResBlock if resblock_updown is True, otherwise use simple Upsample
                    if self.resblock_updown:
                        layers.append(
                            ResBlock(
                                channels=ch, 
                                emb_channels=time_embed_dim, 
                                dropout=dropout, 
                                out_channels=ch, 
                                use_scale_shift_norm=self.use_scale_shift_norm, 
                                up=True # This will perform upsampling inside the ResBlock
                            )
                        )
                    else:
                        layers.append(Upsample(channels=ch, use_conv=True, out_channels=ch))
                    ds //= 2
                    
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # 7. Output Blocks
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)),
        )

    def forward(self, x, timesteps, y = None):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            timesteps (torch.Tensor): 1-D timestep tensor of shape (B,)
            y (torch.Tensor, optional): 1-D class label tensor of shape (B,). Required if num_classes is set.
            
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        # 1. Generate and project Time Embedding
        emb = get_timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(emb)

        # 2. Add Class Embedding if applicable
        if self.num_classes is not None:
            if y is None:
                y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            emb = emb + self.label_emb(y)

        hs = []
        h = x

        # 3. Downsampling Pass (Save feature maps for Skip Connections)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        # 4. Middle Pass
        h = self.middle_block(h, emb)

        # 5. Upsampling Pass (Concatenate Skip Connections)
        for module in self.output_blocks:
            skip = hs.pop()
            # Concatenate along the channel dimension (dim=1)
            h = torch.cat([h, skip], dim=1)
            h = module(h, emb)

        # 6. Output Pass
        h = self.out(h)
        
        return h