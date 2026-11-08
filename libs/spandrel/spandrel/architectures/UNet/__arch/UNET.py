# UNET.py - U-Net architecture for Spandrel (segmentation/binarization, e.g., scanned docs)
# Adapted from standard U-Net; inferable params like SCUNet.

import torch
import torch.nn as nn
import torch.nn.functional as F
from spandrel.util import store_hyperparameters

@store_hyperparameters()
class UNet(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc: int = 1,          # Input channels (1 for grayscale scans)
        out_nc: int = 1,         # Output channels (1 for binary mask)
        base_channels: int = 64, # Base feature channels (scales per level)
        num_levels: int = 4,     # Encoder/decoder depth
    ):
        super().__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.base_channels = base_channels
        self.num_levels = num_levels

        # Encoder: Down path with conv blocks + pooling
        self.enc_layers = nn.ModuleList()
        prev_ch = in_nc
        for level in range(num_levels):
            ch = base_channels * (2 ** level)
            self.enc_layers.append(self._conv_block(prev_ch, ch, pool=True))
            prev_ch = ch

        # Bottleneck: Deeper convs
        bottleneck_ch = base_channels * (2 ** (num_levels - 1))
        self.bottleneck = self._conv_block(bottleneck_ch, bottleneck_ch * 2, pool=False)

        # Decoder: Up path with transposes + skip conv blocks
        self.dec_upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for level in reversed(range(num_levels)):
            ch = base_channels * (2 ** level)
            up_ch = base_channels * (2 ** (level + 1)) if level < num_levels - 1 else bottleneck_ch * 2
            self.dec_upconvs.append(nn.ConvTranspose2d(up_ch, ch, kernel_size=2, stride=2))
            self.dec_blocks.append(self._conv_block(ch * 2, ch, pool=False))  # *2 for skip concat

        # Final output: 1x1 conv + sigmoid for mask
        self.final_conv = nn.Conv2d(base_channels, out_nc, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_ch: int, out_ch: int, pool: bool = True) -> nn.Module:
        """Double conv + ReLU block (pool if encoder). Like SCUNet's conv helpers."""
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        if pool:
            block.add_module('pool', nn.MaxPool2d(2))
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: Encode (skips) -> Bottleneck -> Decode (concat skips) -> Sigmoid mask."""
        # Encode: Collect skips (like SCUNet's down paths)
        skips = []
        for enc in self.enc_layers:
            x = enc(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(skips[-1])  # Use last skip as input

        # Decode: Upsample + concat skips (reverse, like SCUNet's up paths)
        for upconv, dec_block in zip(self.dec_upconvs, self.dec_blocks):
            x = upconv(x)
            skip = skips.pop()
            # Resize skip if shapes mismatch (edge case)
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        # Output mask (0-1 for binarization; threshold >0.5 externally)
        x = self.final_conv(x)
        return self.sigmoid(x)