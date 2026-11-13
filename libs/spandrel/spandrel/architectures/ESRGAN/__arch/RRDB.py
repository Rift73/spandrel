#!/usr/bin/env python3
# type: ignore
from __future__ import annotations

import math

import torch
import torch.nn as nn

from spandrel.__helpers.rocm.common import plain_inference
from spandrel.util import store_hyperparameters

from ...__arch_helpers import block as B
from ...__arch_helpers.inference import banded_tail, tail_scale_and_halo, traced
from ...__arch_helpers.padding import pad_to_multiple

# The high-resolution tail (upsampling, HRconv, conv_last) runs in bands of a BAND_DIVISOR-th
# of the trunk's rows, at least MIN_BAND_ROWS; a trunk of WHOLE_TAIL_ROWS rows or fewer runs it
# whole, where bands only add launches. Its two 64-channel tensors at the output's size set
# the peak otherwise. A 4x model at 1920x1080 in FP16 on an RTX 5090: 7.97 -> 2.28 GiB
# allocated (4.0 -> 1.15 KiB per input pixel), 13.45 -> 3.56 GiB reserved, in the same time;
# bands of a fourth reserved 7.60 GiB, of a sixteenth the same as an eighth.
BAND_DIVISOR = 8
MIN_BAND_ROWS = 32
WHOLE_TAIL_ROWS = 256


@store_hyperparameters()
class RRDBNet(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc: int = 3,
        out_nc: int = 3,
        num_filters: int = 64,
        num_blocks: int = 23,
        scale: int = 4,
        plus: bool = False,
        shuffle_factor: int | None = None,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: B.ConvMode = "CNA",
    ) -> None:
        """
        ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks.
        By Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao,
        and Chen Change Loy.
        This is old-arch Residual in Residual Dense Block Network and is not
        the newest revision that's available at github.com/xinntao/ESRGAN.
        This is on purpose, the newest Network has severely limited the
        potential use of the Network with no benefits.
        This network supports model files from both new and old-arch.
        Args:
            norm: Normalization layer
            act: Activation layer
            upsampler: Upsample layer. upconv, pixel_shuffle
            mode: Convolution mode
        """
        super().__init__()

        self.shuffle_factor = shuffle_factor
        self.scale = scale

        upsample_block = {
            "upconv": B.upconv_block,
            "pixel_shuffle": B.pixelshuffle_block,
        }.get(upsampler)
        if upsample_block is None:
            raise NotImplementedError(f"Upsample mode [{upsampler}] is not found")

        if scale == 3:
            upsample_blocks = upsample_block(
                in_nc=num_filters,
                out_nc=num_filters,
                upscale_factor=3,
                act_type=act,
            )
        else:
            upsample_blocks = [
                upsample_block(
                    in_nc=num_filters,
                    out_nc=num_filters,
                    act_type=act,
                )
                for _ in range(int(math.log(scale, 2)))
            ]

        self.model = B.sequential(
            # fea conv
            B.conv_block(
                in_nc=in_nc,
                out_nc=num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=None,
            ),
            B.ShortcutBlock(
                B.sequential(
                    # rrdb blocks
                    *[
                        B.RRDB(
                            nf=num_filters,
                            kernel_size=3,
                            gc=32,
                            stride=1,
                            bias=True,
                            pad_type="zero",
                            norm_type=norm,
                            act_type=act,
                            mode="CNA",
                            plus=plus,
                        )
                        for _ in range(num_blocks)
                    ],
                    # lr conv
                    B.conv_block(
                        in_nc=num_filters,
                        out_nc=num_filters,
                        kernel_size=3,
                        norm_type=norm,
                        act_type=None,
                        mode=mode,
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            B.conv_block(
                in_nc=num_filters,
                out_nc=num_filters,
                kernel_size=3,
                norm_type=None,
                act_type=act,
            ),
            # hr_conv1
            B.conv_block(
                in_nc=num_filters,
                out_nc=out_nc,
                kernel_size=3,
                norm_type=None,
                act_type=None,
            ),
        )

    def forward(self, x):
        # cuDNN's convolutions take channels last faster and with half the memory, the same
        # values; ROCm's dense-block kernels and exported graphs keep channels first
        channels_last = x.is_cuda and torch.version.cuda is not None and not traced()
        if self.shuffle_factor:
            true_scale = self.scale // self.shuffle_factor
            _, _, h, w = x.size()
            x = pad_to_multiple(x, self.shuffle_factor, mode="reflect")
            x = torch.pixel_unshuffle(x, downscale_factor=self.shuffle_factor)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        plan = self.tail_plan(x)
        if plan is None:
            x = self.model(x)
        else:
            x = self.model[1](self.model[0](x))
            trunk_rows = x.shape[2]
            rows = (
                trunk_rows
                if trunk_rows <= WHOLE_TAIL_ROWS
                # an 8x tail's band holds four times a 4x one's pixels per row
                else max(MIN_BAND_ROWS, trunk_rows // (BAND_DIVISOR * max(1, plan[0] // 4) ** 2))
            )
            x = banded_tail(self.model[2:], x, *plan, rows=rows)
        if self.shuffle_factor:
            x = x[:, :, : h * true_scale, : w * true_scale]
        return x.contiguous() if channels_last else x

    def tail_plan(self, x: torch.Tensor) -> tuple[int, int] | None:
        """(scale, halo) for banding the high-resolution tail in plain eager inference on an
        NVIDIA GPU (any precision); None for the whole pass: tracing, export, training,
        ROCm, the CPU, a tail of no upscaling, or one tail_scale_and_halo cannot band."""
        if not (x.is_cuda and torch.version.hip is None and plain_inference(x, self)):
            return None
        plan = tail_scale_and_halo(self.model[2:])
        return plan if plan is not None and plan[0] > 1 else None
