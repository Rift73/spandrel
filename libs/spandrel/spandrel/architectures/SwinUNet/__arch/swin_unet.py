"""nunif waifu2x SwinUNet (V1) network for Spandrel.

Reproduces ``waifu2x/models/swin_unet.py`` of https://github.com/nagadomi/nunif
(the "new waifu2x"), classes ``SwinUNet``/``SwinUNet2x``/``SwinUNet4x``/``SwinUNet8x``
(``waifu2x.swin_unet_{1x,2x,4x,8x}``) with module and parameter names identical
to the original, so the released checkpoints
(``swin_unet/{art,photo,art_scan}/*.pth``, format ``{"nunif_model": 1, "name": ...,
"kwargs": ..., "state_dict": ...}``) load strictly, without key mapping.

Architecture (all tensor-flow is BHWC except the stem and the very end):

    patch    : Conv(3->C/2, 3x3) - LeakyReLU(0.1) - Conv(C/2->C, 3x3) - LeakyReLU(0.1)   [BCHW]
               then crop 6 px on each side (-6 pad), i.e. the net consumes H-16
    swin1    : L  Swin-Transformer blocks (torchvision Swin V1), C channels, window 6x6
    down1    : Conv 2x2 stride 2, C -> 2C
    swin2    : L  blocks, 2C
    down2    : Conv 2x2 stride 2, 2C -> 2C
    swin3    : 3L blocks, 2C (the "trunk")
    up2      : Linear(2C -> 8C) + PixelShuffle(2)  -> 2C
    (+ skip x4, proj1 = Identity)
    swin4    : L blocks, 2C
    up1      : Linear(2C -> 4C/8C) + PixelShuffle(2) -> C (1x/2x) or 2C (4x/8x)
    (+ skip x3, proj2 = Identity for 1x/2x, Linear(C -> 2C) for 4x/8x)
    swin5    : L blocks
    to_image : Linear(C/2C -> 3*scale^2) + PixelShuffle(scale)    [back to BCHW]

Swin block details (torchvision ``models.swin_transformer`` V1 semantics):
    - norm1/norm2 are ``nn.Identity`` (nunif trains without LayerNorm),
    - ``mlp`` = Linear(dim->2*dim) - GELU - Linear(2*dim->dim) (mlp_ratio=2),
    - alternating regular / shifted (3 px) window attention, window 6x6,
    - relative position bias table ``(2*6-1)^2 x heads`` plus a
      ``relative_position_index`` buffer,
    - attention logits scaled by ``(C // heads) ** -0.5``, shift mask -100.0.

The Swin block is re-implemented here (not imported from torchvision) so the
plugin is self-contained; it is numerically identical to torchvision 0.13+.

Input-size contract:
    The unet requires ``(H - 16) % 48 == 0`` (both 12 and 16 divide H-16),
    e.g. 64, 112, 160, 208, 256, ... and outputs ``scale * (H - 16)``.
    nunif's tile renderer feeds 16+48k tiles that already include 8 px of
    context on every side; the full-image path (``simple_render``) replicate-pads
    the image by ``i2i_offset / scale = 8`` px. This implementation wraps the unet
    so that *any* input size works: the input is replicate-padded to the next
    valid size (always at least 8 px per side) and the output is cropped so that
    ``out = scale * in``. For inputs whose size is a multiple of 48 the padding is
    exactly 8 px and the result is bit-identical to ``simple_render`` of nunif.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ShiftedWindowAttentionV1",
    "SwinTransformerBlockV1",
    "SwinTransformerBlocks",
    "PatchDown",
    "PatchUp",
    "ToImage",
    "SwinUNetBase",
    "SwinUNet",
]


def _next_valid_size(size: int) -> int:
    """Smallest valid unet input size >= size + 16 (nunif contract: 8 px context)."""
    target = size + 16
    k = max(1, -((16 - target) // 48))  # ceil((target - 16) / 48)
    return 16 + 48 * k


class ShiftedWindowAttentionV1(nn.Module):
    """torchvision ``ShiftedWindowAttention`` (Swin V1), self-contained.

    Parameter/buffer names match torchvision: ``qkv``, ``proj``,
    ``relative_position_bias_table``, ``relative_position_index``.
    Input/output layout: BHWC.
    """

    def __init__(self, dim: int, window_size: Sequence[int], shift_size: Sequence[int],
                 num_heads: int, attention_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.window_size = list(window_size)
        self.shift_size = list(shift_size)
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

        # table of relative position biases: (2*Wh-1)*(2*Ww-1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))

        # pair-wise relative position index for each token inside a window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # N*N
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        N = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index]  # N*N, heads
        bias = bias.view(N, N, -1).permute(2, 0, 1).contiguous().unsqueeze(0)  # 1, heads, N, N
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        ws_h, ws_w = self.window_size

        # pad feature maps to multiples of the window size (never hit for
        # valid nunif tile sizes, which are multiples of 6)
        pad_r = (ws_w - W % ws_w) % ws_w
        pad_b = (ws_h - H % ws_h) % ws_h
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        shift_h, shift_w = self.shift_size
        if ws_h >= pad_H:
            shift_h = 0
        if ws_w >= pad_W:
            shift_w = 0

        if shift_h > 0 or shift_w > 0:
            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))

        num_windows = (pad_H // ws_h) * (pad_W // ws_w)
        N = ws_h * ws_w
        x = x.view(B, pad_H // ws_h, ws_h, pad_W // ws_w, ws_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, N, C)

        qkv = F.linear(x, self.qkv.weight, self.qkv.bias)
        qkv = qkv.reshape(x.size(0), x.size(1), 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (C // self.num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
        attn = attn + self.get_relative_position_bias()

        if shift_h > 0 or shift_w > 0:
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = ((0, -ws_h), (-ws_h, -shift_h), (-shift_h, None))
            w_slices = ((0, -ws_w), (-ws_w, -shift_w), (-shift_w, None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0]:h[1], w[0]:w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // ws_h, ws_h, pad_W // ws_w, ws_w)
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, N)
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.attention_dropout, training=self.training)

        x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        x = F.linear(x, self.proj.weight, self.proj.bias)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(B, pad_H // ws_h, pad_W // ws_w, ws_h, ws_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        if shift_h > 0 or shift_w > 0:
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))

        return x[:, :H, :W, :].contiguous()


class SwinTransformerBlockV1(nn.Module):
    """torchvision ``SwinTransformerBlock`` V1 with nunif's Identity norms."""

    def __init__(self, dim: int, num_heads: int, window_size: Sequence[int],
                 shift_size: Sequence[int], mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = ShiftedWindowAttentionV1(dim, window_size, shift_size, num_heads)
        self.norm2 = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SwinTransformerBlocks(nn.Module):
    """nunif's ``SwinTransformerBlocks``: a Sequential of alternating W/SW blocks."""

    def __init__(self, in_channels: int, num_head: int, num_layers: int,
                 window_size: Sequence[int]):
        super().__init__()
        layers = []
        for i_layer in range(num_layers):
            layers.append(
                SwinTransformerBlockV1(
                    in_channels,
                    num_head,
                    window_size=window_size,
                    shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                    mlp_ratio=2.0,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDown(nn.Module):
    """Conv 2x2 stride 2, operates on BHWC (like the original)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC
        return x


class PatchUp(nn.Module):
    """Linear -> PixelShuffle(2), operates on BHWC (like the original)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels, out_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        x = F.pixel_shuffle(x, 2)
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC
        return x


class ToImage(nn.Module):
    """Head: Linear (1x/2x/4x) or Linear-LeakyReLU-Linear (8x) + PixelShuffle."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        super().__init__()
        assert scale_factor in {1, 2, 4, 8}
        self.scale_factor = scale_factor
        self.out_channels = out_channels
        if scale_factor == 1:
            self.proj = nn.Linear(in_channels, out_channels)
        elif scale_factor in {2, 4}:
            self.proj = nn.Linear(in_channels, out_channels * scale_factor ** 2)
        else:  # 8x
            self.proj = nn.Sequential(
                nn.Linear(in_channels, out_channels * scale_factor ** 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(out_channels * scale_factor ** 2, out_channels * scale_factor ** 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
        if self.scale_factor > 1:
            x = F.pixel_shuffle(x, self.scale_factor)
        return x


class SwinUNetBase(nn.Module):
    """``SwinUNetBase`` of nunif (swin_unet.py). BCHW in, BCHW out.

    Output size = ``scale_factor * (H - 16)`` for valid inputs
    (``(H - 16) % 48 == 0``).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_dim: int = 96, base_layers: int = 2, scale_factor: int = 2):
        super().__init__()
        assert scale_factor in {1, 2, 4, 8}
        assert base_dim % 16 == 0 and base_dim % 6 == 0
        assert base_layers % 2 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        C = base_dim
        heads = C // 16
        L = base_layers
        W = [6, 6]
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, C // 2, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(C // 2, C, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.swin1 = SwinTransformerBlocks(C, num_head=heads, num_layers=L, window_size=W)
        self.down1 = PatchDown(C, C * 2)
        self.swin2 = SwinTransformerBlocks(C * 2, num_head=heads, num_layers=L, window_size=W)
        self.down2 = PatchDown(C * 2, C * 2)
        self.swin3 = SwinTransformerBlocks(C * 2, num_head=heads, num_layers=L * 3, window_size=W)
        if scale_factor in {1, 2}:
            self.proj1 = nn.Identity()
            self.up2 = PatchUp(C * 2, C * 2)
            self.proj2 = nn.Identity()
            self.swin4 = SwinTransformerBlocks(C * 2, num_head=heads, num_layers=L, window_size=W)
            self.up1 = PatchUp(C * 2, C)
            self.swin5 = SwinTransformerBlocks(C, num_head=heads, num_layers=L, window_size=W)
            self.to_image = ToImage(C, out_channels, scale_factor=scale_factor)
        elif scale_factor in {4, 8}:
            self.proj1 = nn.Identity()
            self.up2 = PatchUp(C * 2, C * 2)
            self.proj2 = nn.Linear(C, C * 2)
            self.swin4 = SwinTransformerBlocks(C * 2, num_head=heads, num_layers=L, window_size=W)
            self.up1 = PatchUp(C * 2, C * 2)
            self.swin5 = SwinTransformerBlocks(C * 2, num_head=heads, num_layers=L, window_size=W)
            self.to_image = ToImage(C * 2, out_channels, scale_factor=scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.patch(x)
        x2 = F.pad(x2, (-6, -6, -6, -6))
        assert x2.shape[2] % 12 == 0 and x2.shape[2] % 16 == 0, \
            f"input size {x2.shape[2] + 16} is not a valid waifu2x SwinUNet size"
        x2 = x2.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC

        x3 = self.swin1(x2)
        x4 = self.down1(x3)
        x4 = self.swin2(x4)
        x5 = self.down2(x4)
        x5 = self.swin3(x5)
        x5 = self.up2(x5)
        x = x5 + self.proj1(x4)
        x = self.swin4(x)
        x = self.up1(x)
        x = x + self.proj2(x3)
        x = self.swin5(x)
        x = self.to_image(x)
        return x


class SwinUNet(nn.Module):
    """Inference wrapper around :class:`SwinUNetBase` with nunif's size contract.

    Any input size works: the input is replicate-padded to the next valid unet
    size (>= 8 px of context per side, like nunif's tiling) and the output is
    cropped so that ``out = scale * in``. Eval-mode outputs are clamped to
    [0, 1] exactly like the original.

    The submodule is named ``unet`` so state_dict keys are identical to the
    original checkpoints (``unet.*``).
    """

    def __init__(self, unet: SwinUNetBase, scale: int):
        super().__init__()
        self.unet = unet
        self.scale = scale

    @staticmethod
    def _pad_amount(size: int) -> tuple[int, int, int]:
        """Returns (model_size, pad_top, pad_bottom)."""
        model_size = _next_valid_size(size)
        total = model_size - size
        top = total // 2
        return model_size, top, total - top

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        mh, pt, pb = self._pad_amount(H)
        mw, pl, pr = self._pad_amount(W)
        x = F.pad(x, (pl, pr, pt, pb), mode="replicate")
        z = self.unet(x)
        # output row r of the unet corresponds to input row r/scale + 8:
        # crop the borders that only exist because of the padding
        cy = self.scale * max(0, pt - 8)
        cx = self.scale * max(0, pl - 8)
        oh, ow = self.scale * H, self.scale * W
        z = z[..., cy:cy + oh, cx:cx + ow]
        if not self.training:
            z = torch.clamp(z, 0, 1)
        return z
