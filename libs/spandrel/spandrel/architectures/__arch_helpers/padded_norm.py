"""LayerNorm over the first channels of a wider last dimension, as one Triton kernel for
eager inference.

A trunk padded to a multiple of 8 channels (a 180-wide HAT at 184) holds exact zeros in
its last channels; its LayerNorms must take their statistics over the real channels
only and write zeros on the rest. PyTorch has no such operation: built from its own
(slices, FP32 casts, reductions), it took 7.21 ms against F.layer_norm's 1.06 ms at a
HAT-L 1024 input's 1,048,576 tokens on an RTX 5090. Here one pass reads each token's
channels once, computes the mean and variance of the real ones in FP32, applies the
weight and bias and writes the padding channels as zeros, with no intermediate tensor.

Triton is optional: without it, AVAILABLE is False and PaddedLayerNorm computes the same
values with PyTorch's operations.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F

from spandrel.__helpers.rocm.common import from_triton, plain_inference

try:
    import triton
    import triton.language as tl
except ImportError:
    AVAILABLE = False
else:
    AVAILABLE = True

logger = logging.getLogger("spandrel.normalization")

# A kernel that fails to compile or launch turns this off for the rest of the process.
PADDED_NORM = AVAILABLE

if AVAILABLE:

    @triton.jit
    def _padded_layer_norm(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows: tl.int32,
        row_stride,
        real: tl.int32,
        width: tl.int32,
        eps: tl.float32,
        block_rows: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """One program: block_rows tokens. x (rows, width) with a contiguous last
        dimension and row_stride between rows; out (rows, width) contiguous; weight and
        bias (real,)."""
        row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        c = tl.arange(0, block_c)
        in_row = row[:, None] < rows
        is_real = c[None, :] < real
        x = tl.load(
            x_ptr + row[:, None] * row_stride + c[None, :], in_row & is_real, 0
        ).to(tl.float32)
        mean = tl.sum(x, 1) / real
        centered = tl.where(is_real, x - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, 1) / real
        rstd = 1 / tl.sqrt(variance + eps)
        weight = tl.load(weight_ptr + c, c < real, 0).to(tl.float32)
        bias = tl.load(bias_ptr + c, c < real, 0).to(tl.float32)
        y = tl.where(
            is_real, centered * rstd[:, None] * weight[None, :] + bias[None, :], 0.0
        )
        tl.store(
            out_ptr + row[:, None] * width + c[None, :],
            y.to(out_ptr.dtype.element_ty),
            in_row & (c[None, :] < width),
        )


def launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    block_rows: int = 16,
    num_warps: int = 4,
) -> torch.Tensor:
    """The kernel's output, x's shape and type, for x (..., width) whose leading
    dimensions view as rows (the last dimension contiguous) and weight, bias (real,)."""
    width, real = x.shape[-1], weight.shape[0]
    rows = x.view(-1, width)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    count = rows.shape[0]
    with torch.cuda.device(x.device):
        _padded_layer_norm[(triton.cdiv(count, block_rows),)](
            rows,
            weight,
            bias,
            out,
            count,
            rows.stride(0),
            real,
            width,
            eps,
            block_rows=block_rows,
            block_c=max(16, 1 << (width - 1).bit_length()),
            num_warps=num_warps,
        )
    return out


def usable(x: torch.Tensor, module: nn.Module) -> bool:
    """Whether the kernel may run module's normalization on x: plain eager inference
    (plain_inference) on an NVIDIA GPU (a ROCm build keeps the PyTorch path), FP16 or
    BF16, the last dimension contiguous and the leading ones viewable as rows."""
    if not (
        PADDED_NORM
        and x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.stride(-1) == 1
        and plain_inference(x, module)
    ):
        return False
    try:
        x.view(-1, x.shape[-1])
    except RuntimeError:
        return False
    return True


class PaddedLayerNorm(nn.Module):
    """A LayerNorm of `source`'s channels over the first of `padded` channels: statistics
    of the real channels, their weight and bias, zeros on the rest. In plain eager CUDA
    inference it runs the kernel; otherwise, and after a Triton failure (which switches
    the kernel off for the process with one warning), F.layer_norm on the real channels
    followed by the zero channels."""

    def __init__(self, source: nn.LayerNorm, padded: int) -> None:
        super().__init__()
        (real,) = source.normalized_shape
        if source.weight is None or source.bias is None or padded < real:
            raise ValueError(
                "an affine LayerNorm over its last dimension, padded wider"
            )
        self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        self.eps = source.eps
        self.real = real
        self.padded = padded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global PADDED_NORM
        if usable(x, self):
            try:
                return launch(x, self.weight, self.bias, self.eps)
            except Exception as error:
                if not from_triton(error):
                    raise
                PADDED_NORM = False
                logger.warning(
                    "The Triton padded LayerNorm failed (%s: %s); using PyTorch's",
                    type(error).__name__,
                    error,
                )
        normed = F.layer_norm(
            x[..., : self.real], (self.real,), self.weight, self.bias, self.eps
        )
        return F.pad(normed, (0, self.padded - self.real))
