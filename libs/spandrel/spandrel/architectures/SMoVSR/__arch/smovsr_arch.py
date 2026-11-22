from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing_extensions import Self

from spandrel.util import store_hyperparameters

SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "pa_up",
]


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel=3,
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

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


class PA(nn.Module):
    def __init__(self, dim, rep) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvNXC(dim, dim, (1, 1)) if rep else DOConv2d(dim, dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x.mul(self.conv(x))


class UniUpsampleV4_light(nn.Sequential):
    def __init__(
        self,
        upsample: SampleMods = "pa_up",
        scale: int = 2,
        in_dim: int = 24,
        out_dim: int = 3,
        mid_dim: int = 24,
        group: int = 4,
        dysample_end_kernel=1,
        rep: bool = False,
    ) -> None:
        m = []

        if scale == 1 or upsample == "conv":
            m.append(
                ConvNXC(in_dim, out_dim, (3, 3))
                if rep
                else DOConv2d(in_dim, out_dim, 3)
            )
        elif upsample == "pixelshuffledirect":
            m.extend(
                [
                    ConvNXC(in_dim, out_dim * scale**2, (3, 3))
                    if rep
                    else DOConv2d(in_dim, out_dim * scale**2, 3),
                    nn.PixelShuffle(scale),
                ]
            )
        elif upsample == "pixelshuffle":
            m.extend(
                [
                    ConvNXC(in_dim, mid_dim, (3, 3))
                    if rep
                    else DOConv2d(in_dim, mid_dim, 3),
                    nn.LeakyReLU(inplace=True),
                ]
            )
            if (scale & (scale - 1)) == 0:  # scale = 2^n
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            ConvNXC(mid_dim, mid_dim * 4, (3, 3))
                            if rep
                            else DOConv2d(mid_dim, 4 * mid_dim, 3),
                            nn.PixelShuffle(2),
                        ]
                    )
            elif scale == 3:
                m.extend(
                    [
                        ConvNXC(mid_dim, mid_dim * 9, (3, 3))
                        if rep
                        else DOConv2d(mid_dim, 9 * mid_dim, 3),
                        nn.PixelShuffle(3),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(
                ConvNXC(mid_dim, out_dim, (3, 3))
                if rep
                else DOConv2d(mid_dim, out_dim, 3)
            )
        elif upsample == "nearest+conv":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        (
                            ConvNXC(in_dim, in_dim, (3, 3))
                            if rep
                            else DOConv2d(in_dim, in_dim, 3),
                            nn.Upsample(scale_factor=2),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    )
                m.extend(
                    (
                        ConvNXC(in_dim, in_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, in_dim, 3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            elif scale == 3:
                m.extend(
                    (
                        ConvNXC(in_dim, in_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, in_dim, 3),
                        nn.Upsample(scale_factor=scale),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ConvNXC(in_dim, in_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, in_dim, 3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(
                ConvNXC(in_dim, out_dim, (3, 3))
                if rep
                else DOConv2d(in_dim, out_dim, 3)
            )
        elif upsample == "dysample":
            if mid_dim != in_dim:
                m.extend(
                    [
                        ConvNXC(in_dim, out_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, mid_dim, 3),
                        nn.LeakyReLU(inplace=True),
                    ]
                )
            m.append(
                DySample(mid_dim, out_dim, scale, group, end_kernel=dysample_end_kernel)
            )

        elif upsample == "pa_up":
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log2(scale))):
                    m.extend(
                        [
                            nn.Upsample(scale_factor=2),
                            ConvNXC(in_dim, mid_dim, (3, 3))
                            if rep
                            else DOConv2d(in_dim, mid_dim, 3),
                            PA(mid_dim, rep),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            ConvNXC(mid_dim, mid_dim, (3, 3))
                            if rep
                            else DOConv2d(mid_dim, mid_dim, 3),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ]
                    )
                    in_dim = mid_dim
            elif scale == 3:
                m.extend(
                    [
                        nn.Upsample(scale_factor=3),
                        ConvNXC(in_dim, mid_dim, (3, 3))
                        if rep
                        else DOConv2d(in_dim, mid_dim, 3),
                        PA(mid_dim, rep),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ConvNXC(mid_dim, mid_dim, (3, 3))
                        if rep
                        else DOConv2d(mid_dim, mid_dim, 3),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"scale {scale} is not supported. Supported scales: 2^n and 3."
                )
            m.append(
                ConvNXC(mid_dim, out_dim, (3, 3))
                if rep
                else DOConv2d(mid_dim, out_dim, 3)
            )
        else:
            raise ValueError(
                f"An invalid Upsample was selected. Please choose one of {SampleMods}"
            )
        super().__init__(*m)

        self.register_buffer(
            "MetaUpsample",
            torch.tensor(
                [
                    254,
                    list(SampleMods.__args__).index(upsample),
                    scale,
                    in_dim,
                    out_dim,
                    mid_dim,
                    group,
                    int(rep),
                ],
                dtype=torch.uint8,
            ),
        )


def get_same_padding(kernel_size, dilation):
    kh, kw = kernel_size
    dh, dw = dilation

    pad_h = ((kh - 1) * dh) // 2
    pad_w = ((kw - 1) * dw) // 2

    return pad_h, pad_w


def _pair(x):
    if isinstance(x, (tuple, list)):
        if len(x) < 2:
            return x[0], x[0]
        if len(x) > 2:
            return x[0], x[1]
        return x
    return x, x


class DOConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=None,
        groups=1,
        bias=True,
        dilation=1,
        mul=1.0,
    ):
        super().__init__()

        kernel_size = _pair(kernel_size)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.mul = nn.Parameter(torch.ones(1) * mul, requires_grad=True)
        self.dilation = _pair(dilation)
        self.padding = (
            get_same_padding(self.kernel_size, self.dilation)
            if padding is None
            else _pair(padding)
        )
        self.groups = groups
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N
        self.W = Parameter(
            torch.Tensor(out_channels, in_channels // groups, self.D_mul)
        )
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.dow_func = self.dow
        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)
            eye = torch.reshape(
                torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N)
            )
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.d_diag = Parameter(
                    torch.cat([d_diag, zeros], dim=2), requires_grad=False
                )
            else:
                self.d_diag = Parameter(d_diag, requires_grad=False)
            self.dow_func = self.dow_mult
        self.DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
        self.eval_conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        self.forward_func = self.train_forward

    def dow_mult(self):
        D = self.D + self.d_diag
        W = torch.reshape(
            self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul)
        )
        DoW = torch.reshape(torch.einsum("ims,ois->oim", D, W), self.DoW_shape)
        return DoW

    def update_eval(self):
        self.eval_conv.weight.data = (
            (self.dow_func() * self.mul).clone().detach().contiguous()
        )
        self.eval_conv.bias.data = (self.bias * self.mul).clone().detach().contiguous()

    def train(self, mode: bool = True) -> Self:
        if not mode:
            self.update_eval()
            self.forward_func = self.eval_forward
        else:
            self.forward_func = self.train_forward
        return super().train(mode)

    def dow(self):
        return torch.reshape(self.W, self.DoW_shape)

    def fuse_wb(self):
        return self.dow_func(), self.bias

    def train_forward(self, x):
        return F.conv2d(
            x,
            self.dow_func() * self.mul,
            self.bias * self.mul,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def eval_forward(self, x):
        return self.eval_conv(x)

    def forward(self, x):
        return self.forward_func(x)


class ConvNXC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: tuple[int, int] = (3, 3),
        gain1: int = 2,
        mul: int = 1,
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.kernel_size = kernel_size
        self.c_in = c_in
        self.c_out = c_out
        gain = gain1

        self.sk = DOConv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, mul=0.02
        )

        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        self.padding = (pad_h, pad_w)

        self.conv = nn.Sequential(
            DOConv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
            ),
            DOConv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=kernel_size,
                padding=0,
            ),
            DOConv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                mul=mul,
            ),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size, 1, self.padding)
        self.forward_func = self.forward_train

    def update_params(self) -> None:
        self.conv[0].update_eval()
        self.conv[1].update_eval()
        self.conv[2].update_eval()
        w1 = self.conv[0].eval_conv.weight.data.clone().detach()
        w2 = self.conv[1].eval_conv.weight.data.clone().detach()
        w3 = self.conv[2].eval_conv.weight.data.clone().detach()

        kh, kw = self.kernel_size
        pad_h = kh - 1
        pad_w = kw - 1

        w = (
            F.conv2d(
                w1.flip(2, 3).permute(1, 0, 2, 3),
                w2,
                padding=(pad_h, pad_w),
                stride=1,
            )
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )

        self.weight_concat = (
            F.conv2d(
                w.flip(2, 3).permute(1, 0, 2, 3),
                w3,
                padding=0,
                stride=1,
            )
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.sk.update_eval()
        sk_w = self.sk.eval_conv.weight.data.clone().detach()
        b1 = self.conv[0].eval_conv.bias.data.clone().detach()
        b2 = self.conv[1].eval_conv.bias.data.clone().detach()
        b3 = self.conv[2].eval_conv.bias.data.clone().detach()
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3
        sk_b = self.sk.eval_conv.bias.data.clone().detach()

        kh, kw = self.kernel_size
        h_pixels_to_pad = (kh - 1) // 2
        w_pixels_to_pad = (kw - 1) // 2

        sk_w = F.pad(
            sk_w,
            [w_pixels_to_pad, w_pixels_to_pad, h_pixels_to_pad, h_pixels_to_pad],
        )

        self.weight_concat = self.weight_concat + sk_w
        self.eval_conv.weight.data = self.weight_concat.contiguous()

        self.bias_concat = self.bias_concat + sk_b
        self.eval_conv.bias.data = self.bias_concat.contiguous()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            self.update_params()
            self.forward_func = self.forward_eval
        else:
            self.forward_func = self.forward_train
        return self

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = (self.kernel_size[0] - 1) // 2
        pad_w = (self.kernel_size[1] - 1) // 2
        x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
        out = self.conv(x_pad) + self.sk(x)
        return out

    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        return self.eval_conv(x)

    def forward(self, x):
        return self.forward_func(x)


class InceptionDWConv2d(nn.Module):
    """Inception depthweise convolution"""

    def __init__(
        self,
        in_channels: int = 64,
        square_kernel_size: int = 3,
        band_kernel_size: int = 15,
        branch_ratio: int = 0.125,
    ) -> None:
        super().__init__()

        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = DOConv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc
        )
        self.dwconv_w = DOConv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = DOConv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class SMB(nn.Module):
    def __init__(self, in_channels, out_channels=None, rep=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.short = in_channels == out_channels
        self.body = nn.Sequential(
            ConvNXC(in_channels, out_channels, (3, 3))
            if rep
            else DOConv2d(in_channels, out_channels, 3),
            nn.SiLU(True),
            ConvNXC(out_channels, out_channels, (3, 3))
            if rep
            else DOConv2d(out_channels, out_channels, 3),
            nn.SiLU(True),
            ConvNXC(out_channels, out_channels, (3, 3))
            if rep
            else DOConv2d(out_channels, out_channels, 3),
        )
        if self.short:
            self.gate_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1), nn.Tanh()
            )

            with torch.no_grad():
                nn.init.zeros_(self.gate_conv[0].weight)
                self.gate_conv[0].bias.fill_(1.2)
            self.forward = self.forward_short
        else:
            self.forward = self.forwar_wo_short

    def forwar_wo_short(self, x):
        x = self.body(x)
        return F.silu(x, True)

    def forward_short(self, x):
        out = self.body(x)
        return (out + x).mul_(self.gate_conv(out))


class FSMoSR(nn.Module):
    def __init__(
        self,
        dim=24,
        rep=False,
        h_clamp: float | None = None,
        align_kernel: int | None = 7,
    ):
        super().__init__()
        self.blocks_2 = nn.Sequential(*[SMB(dim, rep=rep) for _ in range(3)])
        self.end_block = nn.Sequential(
            SMB(dim, rep=rep),
            ConvNXC(dim, dim, (3, 3)) if rep else DOConv2d(dim, dim, 3),
        )
        self.h_clamp = h_clamp
        self.align = (
            InceptionDWConv2d(dim)
            if align_kernel is None
            else DOConv2d(dim, dim, align_kernel, groups=dim)
        )
        self.z = nn.Sequential(nn.Conv2d(2 * dim, dim, 1), nn.Sigmoid())

    def step_wo_hystory(self, x):
        feat = self.blocks_2(x)
        out = self.end_block(feat + x)
        return out, feat

    def forward(self, x, h):
        return self._step(x, self.align(h))

    def _step(self, x, h):
        if self.h_clamp is not None:
            h = h.clamp(-self.h_clamp, self.h_clamp)
        feat = self.blocks_2(x + h)
        out = self.end_block(feat + x)
        z = self.z(torch.cat([x, h], 1))
        if self.h_clamp is not None:
            feat = feat.clamp(-self.h_clamp, self.h_clamp)
        return out, h + z * (feat - h)


@store_hyperparameters()
class SMoVSR(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_ch: int = 3,
        dim: int = 32,
        rep: bool = True,
        scale: int = 2,
        h_clamp: float | None = None,
        align_kernel: int | None = 7,
    ):
        super().__init__()
        if align_kernel is not None and (
            isinstance(align_kernel, bool)
            or not isinstance(align_kernel, int)
            or align_kernel < 1
            or align_kernel % 2 == 0
        ):
            msg = f"VSMoSR: align_kernel must be None or an odd int >= 1, got {align_kernel!r}"
            raise ValueError(msg)
        self.scale = scale
        self.fe = nn.Sequential(SMB(in_ch, dim, rep=rep), SMB(dim, dim, rep=rep))
        self.f_forward = FSMoSR(dim, rep, h_clamp, align_kernel)
        self.f_backward = FSMoSR(dim, rep, h_clamp, align_kernel)
        self.backward_cat = nn.Conv2d(dim * 2, dim, 1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.PReLU(dim),
            UniUpsampleV4_light(
                "pixelshuffledirect",
                scale,
                dim,
                3,
                32,
                4,
                3,
                rep,
            ),
        )

    def forward(self, x):
        if x.dim() == 4:
            # a single image: a clip of one frame
            return self(x.unsqueeze(1)).squeeze(1)
        b, t, c, h, w = x.size()

        x_flat = x.view(-1, c, h, w)
        short = F.interpolate(
            x_flat, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        feats = self.fe(x_flat)

        feats = feats.view(b, t, -1, h, w)
        short = short.view(b, t, -1, h * self.scale, w * self.scale)
        forward_feats = []
        out, feat_prop = self.f_forward.step_wo_hystory(feats[:, 0, ...])
        feat_prop = self._init_state(feat_prop)
        forward_feats.append(out)
        for i in range(1, t):
            out, feat_prop = self.f_forward(feats[:, i, ...], feat_prop)
            forward_feats.append(out)
        backward_feats = []
        x_last = self.backward_cat(torch.cat([feats[:, -1], forward_feats[-1]], 1))
        out, feat_prop = self.f_backward.step_wo_hystory(x_last)
        feat_prop = self._init_state(feat_prop)
        backward_feats.insert(0, out)
        for i in range(t - 2, -1, -1):
            out, feat_prop = self.f_backward(
                self.backward_cat(torch.cat([feats[:, i], forward_feats[i]], 1)),
                feat_prop,
            )
            backward_feats.insert(0, out)
        f_fused = torch.cat(
            [torch.stack(forward_feats, 1), torch.stack(backward_feats, 1)], dim=2
        )
        f_fused = self.upsampler(f_fused.flatten(0, 1))
        out = f_fused.view(b, t, -1, h * self.scale, w * self.scale) + short
        return out

    def _init_state(self, feat):
        c = self.f_forward.h_clamp
        return feat if c is None else feat.clamp(-c, c)

    def align_gains(self) -> Tensor:
        """Absolute DC gain of each alignment channel in both directions."""
        dev = self.backward_cat.weight.device.type
        gains = []
        with torch.autocast(device_type=dev, enabled=False):
            for f in (self.f_forward, self.f_backward):
                if isinstance(f.align, InceptionDWConv2d):
                    gains.append(
                        self.backward_cat.weight.new_ones(
                            f.align.split_indexes[0], dtype=torch.float32
                        )
                    )
                    convs = (f.align.dwconv_hw, f.align.dwconv_w, f.align.dwconv_h)
                else:
                    convs = (f.align,)
                for conv in convs:
                    gains.append(
                        conv.dow_func().float().sum((1, 2, 3)) * conv.mul.float()
                    )
            return torch.cat(gains).abs()
