# Based on FDAT: https://github.com/stinkybread/fdat/blob/main/fdat.py
# TFDAT from traiNNer-redux (traiNNer/archs/tfdat_arch.py)
from __future__ import annotations

from math import gcd

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys

from spandrel.__helpers.model_descriptor import StateDict
from spandrel.util import store_hyperparameters

from ...FDAT.__arch.fdat import SampleMods3, SimplifiedResidualGroup, UniUpsampleV3


class LightFlowNet(nn.Module):
    """Optical flow from a source frame to a target frame: three stride-2 convolutions
    and a flow prediction at 1/8 resolution."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch * 2, base_ch, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flow_pred = nn.Conv2d(base_ch * 4, 2, 3, 1, 1)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """The B,2,H,W flow from src to tgt (B,C,H,W each) in pixels."""
        _B, _C, H, W = src.shape
        x = torch.cat([src, tgt], dim=1)
        feat = self.enc(x)
        flow_low = self.flow_pred(feat)
        return (
            F.interpolate(flow_low, size=(H, W), mode="bilinear", align_corners=False)
            * 8.0
        )


def warp_frame(x: Tensor, flow: Tensor) -> Tensor:
    """x (B,C,H,W) warped by flow (B,2,H,W, pixels): bilinear sampling by gathers,
    the coordinates clamped to the frame (border replication).

    The coordinates, the weights and the sum are FP32 whatever x's dtype: FP16 spaces
    its values 0.5 apart from 512 and BF16 1 apart from 128 (2 from 256), which would
    snap the sampling positions and, in BF16, round H - 1 = 479 up past the frame.
    For FP32 inputs these are the source's operations."""
    B, C, H, W = x.shape
    flow = flow.float()

    yy = (
        torch.arange(H, device=x.device, dtype=torch.float32)
        .view(1, 1, H, 1)
        .expand(B, 1, H, W)
    )
    xx = (
        torch.arange(W, device=x.device, dtype=torch.float32)
        .view(1, 1, 1, W)
        .expand(B, 1, H, W)
    )

    new_x = xx + flow[:, 0:1]
    new_y = yy + flow[:, 1:2]

    max_x = torch.tensor(W - 1, device=x.device, dtype=torch.float32)
    max_y = torch.tensor(H - 1, device=x.device, dtype=torch.float32)

    new_x = new_x.clamp(0, max_x)
    new_y = new_y.clamp(0, max_y)

    x0 = new_x.floor()
    y0 = new_y.floor()
    x1 = (x0 + 1).clamp(max=max_x)
    y1 = (y0 + 1).clamp(max=max_y)

    dx = new_x - x0
    dy = new_y - y0

    x0_l = x0.long().squeeze(1)
    y0_l = y0.long().squeeze(1)
    x1_l = x1.long().squeeze(1)
    y1_l = y1.long().squeeze(1)

    x_flat = x.view(B, C, -1)

    idx_00 = (y0_l * W + x0_l).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)
    idx_01 = (y0_l * W + x1_l).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)
    idx_10 = (y1_l * W + x0_l).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)
    idx_11 = (y1_l * W + x1_l).unsqueeze(1).expand(B, C, H, W).reshape(B, C, -1)

    p00 = x_flat.gather(-1, idx_00).view(B, C, H, W)
    p01 = x_flat.gather(-1, idx_01).view(B, C, H, W)
    p10 = x_flat.gather(-1, idx_10).view(B, C, H, W)
    p11 = x_flat.gather(-1, idx_11).view(B, C, H, W)

    w00 = (1 - dx) * (1 - dy)
    w01 = dx * (1 - dy)
    w10 = (1 - dx) * dy
    w11 = dx * dy

    return (w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11).to(x.dtype)


@store_hyperparameters()
class TFDAT(nn.Module):
    """Temporal FDAT for video super-resolution.

    Input: (B, T, C, H, W), a clip of T frames (T odd, clip_size).
    Output: (B, C, H*scale, W*scale), the upscaled centre frame.

    The neighbours are warped to the centre frame by a light optical flow, averaged,
    and fused with the centre frame's features before the FDAT backbone.
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 2,
        clip_size: int = 5,
        embed_dim: int = 64,
        num_groups: int = 4,
        depth_per_group: int = 3,
        num_heads: int = 4,
        window_size: int = 8,
        ffn_expansion_ratio: float = 2.0,
        aim_reduction_ratio: int = 8,
        group_block_pattern: list[str] | None = None,
        mid_dim: int = 64,
        upsampler_type: SampleMods3 = "pixelshuffle",
        flow_base_ch: int = 32,
    ) -> None:
        if group_block_pattern is None:
            group_block_pattern = ["spatial", "channel"]
        if clip_size % 2 == 0:
            raise ValueError("clip_size must be odd")
        super().__init__()

        self.clip_size = clip_size
        self.upscale = scale
        self.num_in_ch = num_in_ch

        # Scales 1 and 2 run at an internal scale of 4 after a pixel unshuffle.
        self.unshuffle = 1
        internal_scale = scale
        if scale <= 2:
            self.unshuffle = 4 // max(scale, 1) if scale > 0 else 4
            internal_scale = 4
            conv_in_ch = num_in_ch * self.unshuffle**2
        else:
            conv_in_ch = num_in_ch

        self.flow_net = LightFlowNet(num_in_ch, flow_base_ch)

        # Without an unshuffle, a shallow stem enriches the few input channels.
        if self.unshuffle == 1:
            self.shallow_stem = nn.Sequential(
                nn.Conv2d(num_in_ch, embed_dim // 2, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(embed_dim // 2, num_in_ch, 3, 1, 1, bias=True),
            )
        else:
            self.shallow_stem = None

        self.conv_first = nn.Conv2d(conv_in_ch, embed_dim, 3, 1, 1, bias=True)

        # The centre frame's features and the averaged aligned neighbours' features
        self.temporal_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True),
        )

        ad = depth_per_group * len(group_block_pattern)
        self.groups = nn.Sequential(
            *[
                SimplifiedResidualGroup(
                    embed_dim,
                    ad,
                    num_heads,
                    window_size,
                    ffn_expansion_ratio,
                    aim_reduction_ratio,
                    group_block_pattern,
                    [0.0] * ad,
                )
                for _ in range(num_groups)
            ]
        )

        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = UniUpsampleV3(
            upsampler_type, internal_scale, embed_dim, num_out_ch, mid_dim, 4
        )

        # The flow net's three stride-2 convolutions need multiples of 8, the
        # windows after the unshuffle multiples of unshuffle * window_size.
        self.align = (
            self.unshuffle * window_size * 8 // gcd(self.unshuffle * window_size, 8)
        )

    def load_state_dict(
        self,
        state_dict: StateDict,
        *args,
        **kwargs,
    ) -> _IncompatibleKeys:
        state_dict["upsampler.MetaUpsample"] = self.upsampler.MetaUpsample
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape
        center_idx = T // 2

        # The flow net, the windows and the unshuffle need multiples of align.
        pad_h = (self.align - H % self.align) % self.align
        pad_w = (self.align - W % self.align) % self.align
        if pad_h > 0 or pad_w > 0:
            x = x.view(B * T, C, H, W)
            # reflect needs a padding smaller than the size
            pad_mode = "reflect" if pad_h < H and pad_w < W else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
            x = x.view(B, T, C, H + pad_h, W + pad_w)
        H_pad, W_pad = H + pad_h, W + pad_w

        center = x[:, center_idx]

        aligned_sum = torch.zeros(B, C, H_pad, W_pad, device=x.device, dtype=x.dtype)
        for t in range(T):
            if t == center_idx:
                continue
            neighbor = x[:, t]
            flow = self.flow_net(neighbor, center)
            aligned = warp_frame(neighbor, flow)
            aligned_sum = aligned_sum + aligned

        if self.unshuffle > 1:
            center = F.pixel_unshuffle(center, self.unshuffle)
            aligned_sum = F.pixel_unshuffle(aligned_sum, self.unshuffle)
        elif self.shallow_stem is not None:
            center = self.shallow_stem(center) + center
            aligned_sum = self.shallow_stem(aligned_sum) + aligned_sum

        center_feat = self.conv_first(center)
        aligned_feat = self.conv_first(aligned_sum / max(T - 1, 1))

        fused = self.temporal_fuse(torch.cat([center_feat, aligned_feat], dim=1))

        x_shallow = fused
        x_deep = self.groups(x_shallow)
        x_deep = self.conv_after(x_deep)
        x_out = self.upsampler(x_deep + x_shallow)

        return x_out[:, :, : H * self.upscale, : W * self.upscale]
