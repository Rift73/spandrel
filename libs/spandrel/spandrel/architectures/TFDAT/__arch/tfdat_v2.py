# Based on FDAT: https://github.com/stinkybread/fdat/blob/main/fdat.py
# TFDAT v2 (traiNNer-redux tfdat_arch.py, second version)
from __future__ import annotations

from math import gcd

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys

from spandrel.__helpers.model_descriptor import StateDict
from spandrel.util import store_hyperparameters

from ...FDAT.__arch.fdat import SampleMods3, SimplifiedResidualGroup, UniUpsampleV3


def warp_frame(x: Tensor, flow: Tensor) -> Tensor:
    """x (B,C,H,W) warped by flow (B,2,H,W, pixels): bilinear sampling by gathers,
    border-replicated outside the frame.

    The sampling position is an exact integer grid plus the flow's floor, and the
    weights come from the flow's fraction alone, so they keep their precision in
    FP16 at any frame size."""
    B, C, H, W = x.shape

    yy = torch.arange(H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.arange(W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)

    fx = flow[:, 0:1]
    fy = flow[:, 1:2]
    fx0 = fx.floor()
    fy0 = fy.floor()
    dx = fx - fx0
    dy = fy - fy0

    x0 = xx + fx0.long()
    y0 = yy + fy0.long()
    x1 = x0 + 1
    y1 = y0 + 1

    # Min/Max rather than Clip: TensorRT takes them on integers
    zero = torch.zeros(1, dtype=torch.long, device=x.device)
    max_x = torch.full((1,), W - 1, dtype=torch.long, device=x.device)
    max_y = torch.full((1,), H - 1, dtype=torch.long, device=x.device)
    x0 = torch.minimum(torch.maximum(x0, zero), max_x)
    x1 = torch.minimum(torch.maximum(x1, zero), max_x)
    y0 = torch.minimum(torch.maximum(y0, zero), max_y)
    y1 = torch.minimum(torch.maximum(y1, zero), max_y)

    x_flat = x.reshape(B, C, H * W)
    idx_00 = (y0 * W + x0).view(B, 1, H * W).expand(B, C, H * W)
    idx_01 = (y0 * W + x1).view(B, 1, H * W).expand(B, C, H * W)
    idx_10 = (y1 * W + x0).view(B, 1, H * W).expand(B, C, H * W)
    idx_11 = (y1 * W + x1).view(B, 1, H * W).expand(B, C, H * W)

    p00 = x_flat.gather(-1, idx_00).view(B, C, H, W)
    p01 = x_flat.gather(-1, idx_01).view(B, C, H, W)
    p10 = x_flat.gather(-1, idx_10).view(B, C, H, W)
    p11 = x_flat.gather(-1, idx_11).view(B, C, H, W)

    w00 = (1 - dx) * (1 - dy)
    w01 = dx * (1 - dy)
    w10 = (1 - dx) * dy
    w11 = dx * dy

    return w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11


def _flow_head(in_ch: int, hidden_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_ch, hidden_ch, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_ch, 2, 3, 1, 1),
    )


class FlowPyramidNet(nn.Module):
    """Coarse-to-fine optical flow over three levels (SPyNet style).

    A shared encoder builds a 1/2, 1/4, 1/8 feature pyramid per frame. The flow is
    estimated at 1/8 and refined at 1/4 and 1/2, each time from the neighbour's
    features warped by the upsampled flow.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 3
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c3, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # The coarsest level predicts the flow, the finer levels residuals
        self.dec3 = _flow_head(c3 * 2, c3)
        self.dec2 = _flow_head(c2 * 2 + 2, c2)
        self.dec1 = _flow_head(c1 * 2 + 2, c1)

    def encode(self, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        f1 = self.enc1(frame)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        return f1, f2, f3

    @staticmethod
    def _up2(flow: Tensor) -> Tensor:
        return (
            F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=False)
            * 2.0
        )

    def forward(
        self,
        pyr_src: tuple[Tensor, Tensor, Tensor],
        pyr_tgt: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """The full-resolution B,2,H,W flow (pixels) with which warp_frame aligns the
        source (neighbour) frame to the target (centre) frame, from their pyramids."""
        f1s, f2s, f3s = pyr_src
        f1t, f2t, f3t = pyr_tgt

        flow3 = self.dec3(torch.cat([f3s, f3t], dim=1))

        up2 = self._up2(flow3)
        f2s_w = warp_frame(f2s, up2)
        flow2 = up2 + self.dec2(torch.cat([f2s_w, f2t, up2], dim=1))

        up1 = self._up2(flow2)
        f1s_w = warp_frame(f1s, up1)
        flow1 = up1 + self.dec1(torch.cat([f1s_w, f1t, up1], dim=1))

        return self._up2(flow1)


class AlignmentRefineBlock(nn.Module):
    """A residual correction of a warped neighbour's features from them and the
    centre frame's features, shared across the neighbours."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, warped: Tensor, center: Tensor) -> Tensor:
        return warped + self.body(torch.cat([warped, center], dim=1))


class TemporalAttentionFusion(nn.Module):
    """Confidence-gated temporal fusion (TSA style).

    Each aligned neighbour gets a per-pixel confidence in [0, 1] from its embedded
    similarity with the centre frame and its photometric warp error. The centre and
    the gated neighbours are concatenated and fused, then gated by a squeeze-excite.
    """

    def __init__(self, dim: int, clip_size: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.emb_center = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.emb_neighbor = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.conf = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * clip_size, dim, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self, center: Tensor, neighbors: list[Tensor], errs: list[Tensor]
    ) -> Tensor:
        emb_c = self.emb_center(center)
        scale = self.dim**-0.5
        gated = [center]
        for feat, err in zip(neighbors, errs, strict=True):
            sim = (emb_c * self.emb_neighbor(feat)).sum(dim=1, keepdim=True) * scale
            w = torch.sigmoid(self.conf(torch.cat([sim, err], dim=1)))
            gated.append(feat * w)
        f = self.fuse(torch.cat(gated, dim=1))
        return f * self.se(f)


@store_hyperparameters()
class TFDATv2(nn.Module):
    """Temporal FDAT for video super-resolution, second version.

    Input: (B, T, C, H, W), a clip of T = clip_size frames (odd), or (B, T*C, H, W)
    with the frames stacked on the channels.
    Output: (B, C, H*scale, W*scale), the upscaled centre frame.

    A pyramid optical flow aligns each neighbour to the centre frame; every frame is
    featurized on its own, the warped neighbours' features are corrected and fused
    with confidence gates; the FDAT backbone (with shifted windows) and the upsampler
    reconstruct a residual on a bilinear upscale of the centre frame, and a narrow
    convolution refines the result at output resolution.
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
        self.res_connect = num_in_ch == num_out_ch

        # Scales 1 and 2 run at an internal scale of 4 after a pixel unshuffle.
        self.unshuffle = 1
        internal_scale = scale
        if scale <= 2:
            self.unshuffle = 4 // max(scale, 1) if scale > 0 else 4
            internal_scale = 4
            conv_in_ch = num_in_ch * self.unshuffle**2
        else:
            conv_in_ch = num_in_ch

        self.flow_net = FlowPyramidNet(num_in_ch, flow_base_ch)

        # Shared per-frame shallow feature extraction
        self.conv_first = nn.Sequential(
            nn.Conv2d(conv_in_ch, embed_dim, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True),
        )

        self.align_refine = AlignmentRefineBlock(embed_dim)

        self.temporal_fusion = TemporalAttentionFusion(
            embed_dim, clip_size, aim_reduction_ratio
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
                    shifted_windows=True,
                    stable_norm=True,
                )
                for _ in range(num_groups)
            ]
        )

        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False)
        self.upsampler = UniUpsampleV3(
            upsampler_type, internal_scale, embed_dim, num_out_ch, mid_dim, 4
        )

        # A residual cleanup at output resolution
        self.hr_refine = nn.Sequential(
            nn.Conv2d(num_out_ch, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, num_out_ch, 3, 1, 1),
        )

        # The flow pyramid's three stride-2 levels need multiples of 8, the
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

    def _to_feat(self, img: Tensor) -> Tensor:
        if self.unshuffle > 1:
            img = F.pixel_unshuffle(img, self.unshuffle)
        return self.conv_first(img)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            b, tc, h, w = x.shape
            x = x.reshape(b, self.clip_size, tc // self.clip_size, h, w)
        B, T, C, H, W = x.shape
        if self.clip_size != T:
            raise ValueError(f"expected clip_size={self.clip_size} frames, got {T}")
        center_idx = T // 2

        # The flow net, the windows and the unshuffle need multiples of align.
        pad_h = (self.align - H % self.align) % self.align
        pad_w = (self.align - W % self.align) % self.align
        if pad_h > 0 or pad_w > 0:
            x = x.reshape(B * T, C, H, W)
            # reflect needs a padding smaller than the size
            pad_mode = "reflect" if pad_h < H and pad_w < W else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
            x = x.view(B, T, C, H + pad_h, W + pad_w)

        center = x[:, center_idx]

        # The centre frame's pyramid is encoded once for every pair.
        pyr_center = self.flow_net.encode(center)
        warped_neighbors = []
        warp_errs = []
        for t in range(T):
            if t == center_idx:
                continue
            neighbor = x[:, t]
            flow = self.flow_net(self.flow_net.encode(neighbor), pyr_center)
            warped = warp_frame(neighbor, flow)
            warped_neighbors.append(warped)
            # the photometric error: a signal of occlusion or misalignment
            warp_errs.append((warped - center).abs().mean(dim=1, keepdim=True))

        center_feat = self._to_feat(center)
        neighbor_feats = [
            self.align_refine(self._to_feat(w), center_feat) for w in warped_neighbors
        ]

        # The error maps at feature resolution
        if self.unshuffle > 1:
            warp_errs = [F.avg_pool2d(e, self.unshuffle) for e in warp_errs]

        fused = self.temporal_fusion(center_feat, neighbor_feats, warp_errs)

        x_deep = self.groups(fused)
        x_deep = self.conv_after(x_deep)
        out = self.upsampler(x_deep + fused)

        # The residual on the centre frame
        if self.res_connect:
            if self.upscale > 1:
                base = F.interpolate(
                    center,
                    scale_factor=self.upscale,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                base = center
            out = out + base

        out = out + self.hr_refine(out)

        return out[:, :, : H * self.upscale, : W * self.upscale]
