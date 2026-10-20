"""
GFISRV2 Architecture adapted for Spandrel.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.onnx
from torch import Tensor
from torch import nn

import os
import ctypes

try:
    import tensorrt
except ImportError:
    pass

try:
    plugin_path = os.path.join(os.path.dirname(__file__), "trt_dft_plugins.dll")
    if os.path.exists(plugin_path):
        ctypes.CDLL(plugin_path, winmode=0)
except Exception:
    pass

from spandrel.util import store_hyperparameters
# pylint: disable=relative-beyond-top-level
from ...__arch_helpers.padding import pad_to_multiple

# pylint: disable=invalid-name, too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-locals, too-many-statements, too-many-branches
# pylint: disable=too-many-instance-attributes, not-callable, abstract-method
# pylint: disable=arguments-differ

SAMPLE_MODS = (
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "transpose+conv",
    "lda",
    "pa_up",
)


class DySample(nn.Module):
    """
    Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel: int = 1,
    ) -> None:
        super().__init__()

        if in_channels <= groups or in_channels % groups != 0:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(
                in_channels, out_ch, end_kernel, 1, end_kernel // 2
            )
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> Tensor:
        """Initialize the default sampling positions."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of DySample."""
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h) + 0.5
        coords_w = torch.arange(w) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [w, h], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
            .view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


class LayerNorm(nn.Module):
    """Custom LayerNorm implementation."""

    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of LayerNorm."""
        if x.is_contiguous(memory_format=torch.channels_last):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class LdaAqu(nn.Module):
    """Linear Deformable Attention module."""

    def __init__(
        self,
        in_channels: int = 48,
        reduction_factor: int = 4,
        nh: int = 1,
        scale_factor: float = 2.0,
        k_e: int = 3,
        k_u: int = 3,
        n_groups: int = 2,
        range_factor: int = 11,
        rpb: bool = True,
    ) -> None:
        super().__init__()
        self.k_u = k_u
        self.num_head = nh
        self.scale_factor = scale_factor
        self.n_groups = n_groups
        self.offset_range_factor = range_factor

        self.attn_dim = in_channels // (reduction_factor * self.num_head)
        self.scale = self.attn_dim**-0.5
        self.rpb = rpb
        self.hidden_dim = in_channels // reduction_factor
        self.proj_q = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_k = nn.Conv2d(
            in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.group_channel = in_channels // (reduction_factor * self.n_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                self.group_channel,
                self.group_channel,
                3,
                1,
                1,
                groups=self.group_channel,
                bias=False,
            ),
            LayerNorm(self.group_channel),
            nn.SiLU(),
            nn.Conv2d(self.group_channel, 2 * k_u**2, k_e, 1, k_e // 2),
        )

        self.layer_norm = LayerNorm(in_channels)

        self.pad = int((self.k_u - 1) / 2)
        base = np.arange(-self.pad, self.pad + 1).astype(np.float32)
        base_y = np.repeat(base, self.k_u)
        base_x = np.tile(base, self.k_u)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    1, self.num_head, 1, self.k_u**2, self.hidden_dim // self.num_head
                )
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def init_weights(self) -> None:
        """Initialize weights of the module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        if self.conv_offset[-1].bias is not None:
            nn.init.constant_(self.conv_offset[-1].bias, 0)

    def get_offset(self, offset: Tensor, h_out: int, w_out: int) -> Tensor:
        """Compute the final offset for deformable attention."""
        b = offset.shape[0]
        device = offset.device
        row_indices = torch.arange(h_out, device=device)
        col_indices = torch.arange(w_out, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices, indexing="ij")
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(
            1, h_out, w_out, 2
        )

        d = offset.shape[1] // (self.k_u * self.k_u)
        offset = offset.view(b, self.k_u, self.k_u, d, h_out, w_out).permute(0, 1, 4, 2, 5, 3)

        offset = offset + index_tensor.view(1, 1, h_out, 1, w_out, 2)
        offset = offset.contiguous().view(b, self.k_u * h_out, self.k_u * w_out, 2)

        offset[..., 0] = 2 * offset[..., 0] / (h_out - 1) - 1
        offset[..., 1] = 2 * offset[..., 1] / (w_out - 1) - 1
        offset = offset.flip(-1)
        return offset

    def extract_feats(self, x: Tensor, offset: Tensor, ks: int = 3) -> Tensor:
        """Extract features using grid_sample."""
        out = F.grid_sample(
            x, offset, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        b, c, h_ks, w_ks = out.shape
        h = h_ks // ks
        w = w_ks // ks
        out = out.view(b, c, ks, h, ks, w).permute(0, 2, 4, 1, 3, 5).reshape(b, ks * ks, c, h, w)
        return out

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of LdaAqu."""
        b, c, h, w = x.shape
        out_h, out_w = int(h * self.scale_factor), int(w * self.scale_factor)
        v = x
        x = self.layer_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)

        q = F.interpolate(
            q, (out_h, out_w), mode="bilinear", align_corners=True
        )
        q_off = q.view(b * self.n_groups, -1, out_h, out_w)
        pred_offset = self.conv_offset(q_off)
        offset = pred_offset.tanh().mul(self.offset_range_factor) + self.base_offset.to(
            x.dtype
        )

        k = k.view(b * self.n_groups, self.hidden_dim // self.n_groups, h, w)
        v = v.view(b * self.n_groups, c // self.n_groups, h, w)
        offset = self.get_offset(offset, out_h, out_w)
        k = self.extract_feats(k, offset=offset)
        v = self.extract_feats(v, offset=offset)

        c_q = q.shape[1] // self.num_head
        q = q.view(b, self.num_head, c_q, out_h, out_w).permute(0, 1, 3, 4, 2)
        q = q.reshape(b, self.num_head, out_h * out_w, 1, c_q)

        bg, n, c_dim, h_feat, w_feat = k.shape
        b_orig = bg // self.n_groups
        k = k.view(b_orig, self.n_groups, n, c_dim, h_feat, w_feat).permute(0, 4, 5, 2, 1, 3)
        k = k.reshape(b_orig, h_feat * w_feat, n, self.n_groups * c_dim)

        v = v.view(b_orig, self.n_groups, n, c_dim, h_feat, w_feat).permute(0, 4, 5, 2, 1, 3)
        v = v.reshape(b_orig, h_feat * w_feat, n, self.n_groups * c_dim)

        c_k = k.shape[-1] // self.num_head
        k = k.view(b_orig, h_feat * w_feat, n, self.num_head, c_k).permute(0, 3, 1, 2, 4)
        v = v.view(b_orig, h_feat * w_feat, n, self.num_head, c_k).permute(0, 3, 1, 2, 4)

        if self.rpb:
            k = k + self.relative_position_bias_table

        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        t = out.shape[-2]
        c_out = out.shape[-1]
        out = out.view(b_orig, self.num_head, out_h, out_w, t, c_out).permute(0, 1, 5, 4, 2, 3)
        out = out.reshape(b_orig, self.num_head * c_out, t * out_h, out_w)

        return out


class PA(nn.Module):
    """Pixel Attention Module."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of PA."""
        return x.mul(self.conv(x))


class UniUpsampleV3(nn.Sequential):
    """Universal Upsample Module Version 3."""

    def __init__(
        self,
        upsample: str = "pa_up",
        scale: int = 2,
        in_dim: int = 48,
        out_dim: int = 3,
        mid_dim: int = 48,
        group: int = 4,
        dysample_end_kernel: int = 1,
    ) -> None:
        m = []

        if upsample not in SAMPLE_MODS:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SAMPLE_MODS}"
            )

        if scale == 1 or upsample == "conv":
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "pixelshuffledirect":
            m.extend(
                [nn.Conv2d(in_dim, out_dim * scale**2, 3, 1, 1), nn.PixelShuffle(scale)]
            )
        elif upsample == "pixelshuffle":
            m.extend([nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)])
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [nn.Conv2d(mid_dim, 4 * mid_dim, 3, 1, 1), nn.PixelShuffle(2)]
                    )
            elif scale == 3:
                m.extend([nn.Conv2d(mid_dim, 9 * mid_dim, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_dim, in_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
            m.append(nn.Conv2d(in_dim, out_dim, 3, 1, 1))
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(
                DySample(mid_dim, out_dim, scale, group, end_kernel=dysample_end_kernel)
            )
        elif upsample == "transpose+conv":
            if scale == 2:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1))
            elif scale == 3:
                m.append(nn.ConvTranspose2d(in_dim, out_dim, 3, 3, 0))
            elif scale == 4:
                m.extend(
                    [
                        nn.ConvTranspose2d(in_dim, in_dim, 4, 2, 1),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
                    ]
                )
            else:
                raise ValueError(f"scale {scale} is not supported. Supported scales: 2, 3, 4")
            m.append(nn.Conv2d(out_dim, out_dim, 3, 1, 1))
        elif upsample == "lda":
            if mid_dim != in_dim:
                m.extend(
                    [nn.Conv2d(in_dim, mid_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)]
                )
            m.append(LdaAqu(mid_dim, scale_factor=scale))
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))
        elif upsample == "pa_up":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                            PA(mid_dim),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ]
                    )
                    in_dim = mid_dim
            elif scale == 3:
                m.extend(
                    [
                        nn.Upsample(scale_factor=3),
                        nn.Conv2d(in_dim, mid_dim, 3, 1, 1),
                        PA(mid_dim),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(f"scale {scale} is not supported. Supported scales: 2^n and 3.")
            m.append(nn.Conv2d(mid_dim, out_dim, 3, 1, 1))

        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    3,
                    SAMPLE_MODS.index(upsample),
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                ],
                dtype=torch.uint8,
            ),
        )


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    def __init__(self, dim: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones([dim, 1, 1]))
        self.offset = nn.Parameter(torch.zeros([dim, 1, 1]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of RMSNorm."""
        norm_x = x.norm(2, dim=1, keepdim=True)
        d_x = x.size(1)
        rms_x = norm_x * (d_x ** (-1.0 / 2))
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed + self.offset


class CustomRFFT2(torch.autograd.Function):
    """Custom ONNX-compatible Real FFT2 Function (TRT Plugin Stub)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g, x: torch.Value):
        # Важно! Используем домен com.microsoft, как прописано в C++ коде плагина Алексея
        # Нормализация (norm="ortho") в ONNX-стандарте передается как пустой атрибут или решается внутри
        return g.op("com.microsoft::Rfft2", x)


class CustomIRFFT2(torch.autograd.Function):
    """Custom ONNX-compatible Inverse Real FFT2 Function (TRT Plugin Stub)."""

    @staticmethod
    def forward(ctx, x_ri: torch.Tensor):
        x_c = torch.view_as_complex(x_ri)
        return torch.fft.irfft2(x_c, dim=(2, 3), norm="ortho")

    @staticmethod
    def symbolic(g, x: torch.Value):
        return g.op("com.microsoft::Irfft2", x)


class CustomRfft2Wrap(nn.Module):
    """Wrapper for Custom RFFT2."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of CustomRfft2Wrap."""
        if self.training:
            y = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
            return torch.view_as_real(y)
        return CustomRFFT2.apply(x)


class CustomIrfft2Wrap(nn.Module):
    """Wrapper for Custom IRFFT2."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of CustomIrfft2Wrap."""
        if self.training:
            x_c = torch.view_as_complex(x)
            return torch.fft.irfft2(x_c, dim=(2, 3), norm="ortho")
        return CustomIRFFT2.apply(x)


class FourierUnit(nn.Module):
    """Fourier Unit for Global Feature Extraction."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.rn = RMSNorm(out_channels * 2)
        self.post_norm = RMSNorm(out_channels)

        self.fdc = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            bias=True,
        )

        self.fpe = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=in_channels * 2,
            kernel_size=3,
            padding=1,
            groups=in_channels * 2,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.irfft2 = CustomIrfft2Wrap()
        self.rfft2 = CustomRfft2Wrap()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Fourier Unit."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        b, c, h, _ = x.shape
        ffted = self.rfft2(x)
        ffted = ffted.permute(0, 4, 1, 2, 3).contiguous()
        ffted = ffted.view(b, c * 2, h, -1).to(orig_dtype)
        ffted = self.rn(ffted)
        ffted = self.fpe(ffted) + ffted
        ffted = self.fdc(ffted)
        ffted = self.gelu(ffted)
        ffted = ffted.view(b, c, 2, h, -1).permute(0, 1, 3, 4, 2).contiguous().float()
        out = self.irfft2(ffted)
        out = self.post_norm(out.to(orig_dtype))
        return out


class InceptionDWConv2d(nn.Module):
    """Inception depthwise convolution."""

    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
        shift: int = 0,
    ) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)

        # Removed nn.ModuleList to maintain strict compatibility with the original state dict keys
        self.convs = [
            FourierUnit(in_channels - 3 * gc, in_channels - 3 * gc),
            nn.Conv2d(
                gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
            ),
            nn.Conv2d(
                gc,
                gc,
                kernel_size=(1, band_kernel_size),
                padding=(0, band_kernel_size // 2),
                groups=gc,
            ),
            nn.Conv2d(
                gc,
                gc,
                kernel_size=(band_kernel_size, 1),
                padding=(band_kernel_size // 2, 0),
                groups=gc,
            ),
        ]

        self.pconv = self.convs[shift % 4]
        self.dwconv_hw = self.convs[(shift + 1) % 4]
        self.dwconv_w = self.convs[(shift + 2) % 4]
        self.dwconv_h = self.convs[(shift + 3) % 4]

        indexs = [in_channels - 3 * gc, gc, gc, gc]
        self.split_indexes = (
            indexs[shift % 4],
            indexs[(shift + 1) % 4],
            indexs[(shift + 2) % 4],
            indexs[(shift + 3) % 4],
        )
        self.slices = []
        s = 0
        for idx in self.split_indexes:
            self.slices.append((s, s + idx))
            s += idx

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Inception depthwise convolution."""
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (
                self.pconv(x_id),
                self.dwconv_hw(x_hw),
                self.dwconv_w(x_w),
                self.dwconv_h(x_h),
            ),
            dim=1,
        )


class GatedCNNBlock(nn.Module):
    """Gated CNN Block for GFISRV2."""

    def __init__(
        self, dim: int = 64, expansion_ratio: float = 8 / 3, shift: int = 0
    ) -> None:
        super().__init__()
        hidden = int(expansion_ratio * dim)
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]), requires_grad=True)
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.act = nn.SiLU()
        conv_channels = dim
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.conv = InceptionDWConv2d(dim, shift=shift)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Gated CNN Block."""
        shortcut = x
        x = self.norm(x)
        x = self.fc1(x)
        g, i, c = torch.split(x, self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))

        return x * self.gamma + shortcut


@store_hyperparameters()
class GFISRV2(nn.Module):
    """GFISRV2 Architecture implemented for Spandrel."""

    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc: int = 3,
        dim: int = 48,
        expansion_ratio: float = 8 / 3,
        scale: int = 4,
        out_nc: int = 3,
        upsampler: str = "pixelshuffledirect",
        mid_dim: int = 32,
        pixel_unshuffle: bool = False,
        n_blocks: int = 24,
    ) -> None:
        super().__init__()

        self.true_scale = scale

        if pixel_unshuffle and scale in [1, 2]:
            down = 4 // scale
            self.in_to_dim = nn.Sequential(
                nn.PixelUnshuffle(down), nn.Conv2d(in_nc * down * down, dim, 3, 1, 1)
            )
            scale = 4
            self.pad = down * 2
        else:
            self.in_to_dim = nn.Conv2d(in_nc, dim, 3, 1, 1)
            self.pad = 2

        self.gfisr_body = nn.Sequential(
            *[
                GatedCNNBlock(dim, expansion_ratio=expansion_ratio, shift=i)
                for i in range(n_blocks)
            ]
            + [
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                nn.SiLU(True),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
            ]
        )

        self.upscale = UniUpsampleV3(
            upsampler, scale, dim, out_nc, mid_dim, dysample_end_kernel=3
        )

        self.scale = scale

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Override load_state_dict to handle MetaUpsample buffer."""
        if "upscale.MetaUpsample" not in state_dict:
            state_dict["upscale.MetaUpsample"] = self.upscale.MetaUpsample
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of GFISRV2."""
        _, _, h, w = x.shape

        # Uses standard Spandrel arch helper instead of manual modulo padding
        x = pad_to_multiple(x, self.pad, mode="reflect")
        x = self.in_to_dim(x)

        x = self.gfisr_body(x) + x
        out = self.upscale(x)

        return out[:, :, : h * self.true_scale, : w * self.true_scale]