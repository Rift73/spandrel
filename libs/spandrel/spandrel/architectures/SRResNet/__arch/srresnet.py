"""SRResNet network for Spandrel, adapted from https://github.com/delldu/SRResNet.

Source: ``srresnet.py`` of https://github.com/delldu/SRResNet (classes ``_NetG``
and ``_Residual_Block``), a PyTorch implementation of

    "Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network", Ledig et al., arXiv:1609.04802

The adaptation keeps the module and parameter names of the original
implementation, so state dicts trained with the original repository
(``conv_input``, ``residual.{i}.conv1/in1/conv2/in2``, ``conv_mid``,
``bn_mid``, ``upscale4x.{i}``, ``conv_output``) load strictly, without any
key mapping.

Beyond the original fixed 4x RGB model (64 features, 16 residual blocks,
InstanceNorm, LeakyReLU 0.2), this implementation supports the common
variants of this architecture family:

- 2x/4x/8x upscale factors (``upscale2x``/``upscale4x``/``upscale8x`` modules),
- residual blocks with ``BatchNorm2d`` instead of ``InstanceNorm2d``
  (attribute names ``bn1``/``bn2`` instead of ``in1``/``in2``),
- convolutions with bias,
- arbitrary input/output channel counts and residual block counts.

Inference semantics:
    The original inference code (``demo.py``) runs the model without calling
    ``.eval()``, i.e. in training mode. For InstanceNorm layers this means
    per-instance (input) statistics are used for normalization at inference
    time. Old checkpoints (trained with PyTorch <= 0.4, where
    ``track_running_stats`` defaulted to ``True``) additionally store running
    buffers, but those were never used for inference. To reproduce the
    original behavior exactly, the InstanceNorm layers of this implementation
    always normalize with input statistics. Running buffers are only
    registered (and loaded) when the state dict contains them, purely for
    strict state dict compatibility; they are never used nor updated.

This module additionally implements two further members of the SRResNet
family whose state dicts use different module layouts:

- :class:`SRResNetSequential` -- the "Sequential" layout
  (``model.0``, ``model.1.sub.N.res.M``, ``model.2``, ...), i.e. the
  wrapper layout of the ESRGAN-era checkpoints, with plain Conv-LeakyReLU
  residual blocks instead of RRDBs and a single/multi-stage pixel-shuffle
  upsampler. Real-world example: ``2x-Waifaux-NL3-SRResNet.pth``
  (OpenModelDB, "Emulating Waifu2x at Noise Level 3").
- :class:`MSRResNet` -- the "Modified SRResNet" of BasicSR
  (``basicsr/archs/srresnet_arch.py``; keys ``conv_first``, ``body.N``,
  ``upconv1``/``upconv2``, ``conv_hr``, ``conv_last``). Residual blocks
  without normalization (ReLU inside), LeakyReLU(0.1) elsewhere and a
  bilinear base skip; supports 2x/3x/4x.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SRResNet",
    "ResidualBlock",
    "SRResNetSequential",
    "SequentialResidualBlock",
    "SequentialBody",
    "MSRResNet",
    "ResidualBlockNoBN",
]


class InstanceNorm2d(nn.InstanceNorm2d):
    """``nn.InstanceNorm2d`` that always normalizes with input statistics.

    Running buffers are registered iff ``track_running_stats=True``, purely so
    that state dicts of old SRResNet checkpoints (which always contain them)
    load strictly. The buffers are never used for normalization and never
    updated.

    This exactly replicates the inference behavior of the original SRResNet
    code, which effectively always normalizes with per-instance statistics.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.instance_norm(
            input,
            None,  # running_mean (unused)
            None,  # running_var (unused)
            self.weight if self.affine else None,
            self.bias if self.affine else None,
            True,  # use_input_stats
            0.0,  # momentum (no buffer updates)
            self.eps,
        )


def _make_norm(num_features: int, with_running_stats: bool) -> nn.Module:
    return InstanceNorm2d(
        num_features, affine=True, track_running_stats=with_running_stats
    )


class ResidualBlock(nn.Module):
    """Residual block of SRResNet.

    Adapted from ``_Residual_Block`` of https://github.com/delldu/SRResNet.
    The attribute names ``conv1``/``in1``/``relu``/``conv2``/``in2``
    (InstanceNorm flavor, original) and ``conv1``/``bn1``/``relu``/``conv2``/
    ``bn2`` (BatchNorm flavor) match the original implementations, so their
    state dicts load without key mapping.
    """

    def __init__(
        self,
        num_features: int,
        norm: Literal["instance", "batch"] = "instance",
        with_running_stats: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if norm not in ("instance", "batch"):
            raise ValueError(
                f"Invalid norm {norm!r}; must be 'instance' or 'batch'"
            )

        self._use_batchnorm = norm == "batch"

        self.conv1 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        if self._use_batchnorm:
            self.bn1 = nn.BatchNorm2d(num_features)
        else:
            self.in1 = _make_norm(num_features, with_running_stats)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        if self._use_batchnorm:
            self.bn2 = nn.BatchNorm2d(num_features)
        else:
            self.in2 = _make_norm(num_features, with_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out) if self._use_batchnorm else self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) if self._use_batchnorm else self.in2(out)
        return torch.add(out, identity)


class SRResNet(nn.Module):
    """SRResNet generator.

    Adapted from ``_NetG`` of https://github.com/delldu/SRResNet. With the
    default arguments, this class is structurally identical to the original
    ``_NetG`` and its state dict keys are exactly the same.

    Args:
        in_nc: Number of input image channels.
        out_nc: Number of output image channels.
        nf: Number of features (channels inside the network).
        nb: Number of residual blocks.
        upscale: Upscale factor. One of ``1``, ``2``, ``4``, ``8``.
        kernel_size: Kernel size of the first and last convolution
            (9 in the original implementation).
        norm: Normalization of the residual blocks and the middle
            convolution. ``"instance"`` (original) or ``"batch"``.
        with_running_stats: Whether to register running buffers for
            InstanceNorm layers. Only relevant for strict state dict
            compatibility with old checkpoints; see `InstanceNorm2d`.
        bias: Whether convolutions have bias terms.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 16,
        upscale: int = 4,
        kernel_size: int = 9,
        norm: Literal["instance", "batch"] = "instance",
        with_running_stats: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if upscale not in (1, 2, 4, 8):
            raise NotImplementedError(
                f"upscale factor {upscale} is not supported; must be one of 1, 2, 4, 8"
            )

        padding = kernel_size // 2

        self.conv_input = nn.Conv2d(
            in_channels=in_nc,
            out_channels=nf,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = nn.Sequential(
            *[
                ResidualBlock(
                    num_features=nf,
                    norm=norm,
                    with_running_stats=with_running_stats,
                    bias=bias,
                )
                for _ in range(nb)
            ]
        )

        self.conv_mid = nn.Conv2d(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        # the attribute is named `bn_mid` for both norm flavors, matching the
        # original implementation
        if norm == "batch":
            self.bn_mid = nn.BatchNorm2d(nf)
        else:
            self.bn_mid = _make_norm(nf, with_running_stats)

        self.upscale = upscale
        if upscale > 1:
            # stages of [conv, PixelShuffle(2), LeakyReLU], matching the
            # original `upscale4x` module (conv weight indices 0, 3, 6, ...)
            upsampler: list[nn.Module] = []
            for _ in range(upscale.bit_length() - 1):
                upsampler.extend(
                    [
                        nn.Conv2d(
                            in_channels=nf,
                            out_channels=nf * 4,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=bias,
                        ),
                        nn.PixelShuffle(2),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
            setattr(self, f"upscale{upscale}x", nn.Sequential(*upsampler))

        self.conv_output = nn.Conv2d(
            in_channels=nf,
            out_channels=out_nc,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )

        # weight initialization, same as the original implementation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        if self.upscale > 1:
            out = getattr(self, f"upscale{self.upscale}x")(out)
        out = self.conv_output(out)
        return out


# ---------------------------------------------------------------------------
# Variant 2: "Sequential" layout
# (e.g. 2x-Waifaux-NL3-SRResNet.pth, OpenModelDB)
# ---------------------------------------------------------------------------


class SequentialResidualBlock(nn.Module):
    """Residual block of the Sequential-layout SRResNet.

    Matches the state dict layout ``model.1.sub.N.res.{0,2}``: a ``res``
    sequential of [conv, LeakyReLU(0.2), conv] with a plain identity skip
    around it. Unlike the original SRResNet there is no normalization and
    the convolutions have bias terms.
    """

    def __init__(self, num_features: int, bias: bool = True) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(x, self.res(x))


class SequentialBody(nn.Module):
    """Trunk (``model.1``) of the Sequential-layout SRResNet.

    ``sub`` stacks ``num_blocks`` residual blocks followed by a final
    convolution (the analogue of ESRGAN's ``conv_body``), and a global skip
    connection wraps the whole stack. This mirrors the semantics of the
    ESRGAN-era ``model.*`` checkpoint layout, where the trunk is a wrapper
    module holding a ``sub`` sequential.
    """

    def __init__(self, num_features: int, num_blocks: int, bias: bool = True) -> None:
        super().__init__()
        self.sub = nn.Sequential(
            *[
                SequentialResidualBlock(num_features, bias=bias)
                for _ in range(num_blocks)
            ],
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(x, self.sub(x))


class SRResNetSequential(nn.Module):
    """SRResNet with the "Sequential" state dict layout.

    Reconstructed network of checkpoints such as
    ``2x-Waifaux-NL3-SRResNet.pth`` (OpenModelDB, "Emulating Waifu2x at
    Noise Level 3"), whose state dict keys are::

        model.0.{weight,bias}                     first conv (3x3)
        model.1.sub.{0..nb-1}.res.{0,2}...        residual blocks
        model.1.sub.nb.{weight,bias}              trunk conv
        model.{2+3k}.{weight,bias}                upsampler convs
        model.{conv_hr}.{weight,bias}             64->64 conv
        model.{conv_last}.{weight,bias}           nf->out conv

    This is the wrapper layout of the ESRGAN-era checkpoints (compare the
    keys of ESRGAN models: ``model.0``, ``model.1.sub.N.RDB1...``), but
    with plain Conv-LeakyReLU residual blocks instead of RRDBs, no
    normalization, and a single/multi-stage pixel-shuffle upsampler. The
    reconstruction follows the ESRGAN data flow exactly: no activation
    after the first conv, a global skip around the trunk, and
    [conv, PixelShuffle, LeakyReLU(0.2)] upsampling stages.

    Args:
        in_nc: Number of input image channels.
        out_nc: Number of output image channels.
        nf: Number of features (channels inside the network).
        nb: Number of residual blocks in the trunk.
        scale: Upscale factor. One of ``2``, ``3``, ``4``, ``8``.
        bias: Whether convolutions have bias terms.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 23,
        scale: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if scale not in (2, 3, 4, 8):
            raise NotImplementedError(
                f"upscale factor {scale} is not supported; must be one of 2, 3, 4, 8"
            )

        modules: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_nc,
                out_channels=nf,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            SequentialBody(nf, nb, bias=bias),
        ]

        if scale in (2, 3):
            # single stage: [conv(nf -> nf*scale^2), PixelShuffle(scale), act]
            modules.extend(
                [
                    nn.Conv2d(
                        in_channels=nf,
                        out_channels=nf * scale * scale,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                    ),
                    nn.PixelShuffle(scale),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        else:
            # 4x/8x: stages of [conv(nf -> nf*4), PixelShuffle(2), act]
            for _ in range(scale.bit_length() - 1):
                modules.extend(
                    [
                        nn.Conv2d(
                            in_channels=nf,
                            out_channels=nf * 4,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=bias,
                        ),
                        nn.PixelShuffle(2),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )

        modules.extend(
            [
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=out_nc,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                ),
            ]
        )

        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Variant 3: "Modified SRResNet" (MSRResNet) of BasicSR
# (https://github.com/XPixelGroup/BasicSR, basicsr/archs/srresnet_arch.py)
# ---------------------------------------------------------------------------


class ResidualBlockNoBN(nn.Module):
    """``ResidualBlockNoBN`` of BasicSR (``basicsr/archs/arch_util.py``).

    Conv-ReLU-Conv without normalization, with bias, and an identity skip.
    Matches the state dict layout ``body.N.conv1/conv2``.
    """

    def __init__(self, num_feat: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class MSRResNet(nn.Module):
    """Modified SRResNet of BasicSR (``basicsr/archs/srresnet_arch.py``).

    A compacted SRResNet variant with residual blocks without
    normalization ("similar to EDSR"), LeakyReLU(0.1) activations outside
    the residual blocks (ReLU inside them), pixel-shuffle upsampling and a
    bilinear upsampled base skip. Supports 2x, 3x and 4x upscale factors.

    State dict keys: ``conv_first``, ``body.{i}.conv1/conv2``,
    ``upconv1`` (and ``upconv2`` for 4x), ``conv_hr``, ``conv_last``.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 16,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.upscale = upscale

        if upscale not in (2, 3, 4):
            raise NotImplementedError(
                f"upscale factor {upscale} is not supported; must be one of 2, 3, 4"
            )

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlockNoBN(num_feat) for _ in range(num_block)]
        )

        # upsampling
        if upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        else:  # 2x or 3x
            self.upconv1 = nn.Conv2d(num_feat, num_feat * upscale * upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        else:  # 2x or 3x
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(
            x, scale_factor=self.upscale, mode="bilinear", align_corners=False
        )
        return out + base
