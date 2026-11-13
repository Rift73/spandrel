"""HEART's image-wide i-LN as two Triton kernels for eager inference, over the first
channels of a possibly wider last dimension.

i-LN takes one mean and one variance over every token and channel of an image. A trunk
padded to a multiple of 8 channels (180 at 184) holds exact zeros in its last channels:
the statistics must count the real channels only, and the output must be zero on the
rest. The first kernel sums blocks of tokens (in FP32, from an offset, the image's first
value, which keeps the sum of squares from cancelling), PyTorch adds the blocks' sums in a
fixed order, and the second kernel normalizes, applies the weight and bias and writes the
padding channels as zeros. Three launches, against about seven for the PyTorch form.

Triton is optional: without it, AVAILABLE is False and PaddedILN computes the same values
with PyTorch's operations.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from spandrel.__helpers.rocm.common import from_triton, plain_inference

try:
    import triton
    import triton.language as tl
except ImportError:
    AVAILABLE = False
else:
    AVAILABLE = True

logger = logging.getLogger("spandrel.norm")

# A kernel that fails to compile or launch turns this off for the rest of the process.
ILN_KERNEL = AVAILABLE

_TOKENS = 32

if AVAILABLE:

    @triton.jit
    def _iln_sums(
        x_ptr,
        sums_ptr,
        tokens: tl.int32,
        padded: tl.int32,
        real: tl.int32,
        blocks: tl.int32,
        block_t: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """sums[b, block] = (sum, sum of squares) of x - x[b, 0, 0] over block_t tokens'
        real channels."""
        block, image = tl.program_id(0), tl.program_id(1)
        base = x_ptr + image.to(tl.int64) * tokens * padded
        offset = tl.load(base).to(tl.float32)
        t = block * block_t + tl.arange(0, block_t)
        c = tl.arange(0, block_c)
        inside = (t[:, None] < tokens) & (c[None, :] < real)
        x = tl.load(base + t[:, None].to(tl.int64) * padded + c[None, :], inside, 0)
        x = tl.where(inside, x.to(tl.float32) - offset, 0.0)
        out = sums_ptr + (image * blocks + block) * 2
        tl.store(out, tl.sum(tl.sum(x, 1), 0))
        tl.store(out + 1, tl.sum(tl.sum(x * x, 1), 0))

    @triton.jit
    def _iln_apply(
        x_ptr,
        totals_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        std_ptr,
        tokens: tl.int32,
        padded: tl.int32,
        real: tl.int32,
        eps: tl.float32,
        block_t: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """out = (x - mean) / std * weight + bias on the real channels, zeros on the
        rest; std of each image written once."""
        block, image = tl.program_id(0), tl.program_id(1)
        base = x_ptr + image.to(tl.int64) * tokens * padded
        offset = tl.load(base).to(tl.float32)
        count = tokens.to(tl.float32) * real
        shifted_mean = tl.load(totals_ptr + image * 2) / count
        var = tl.maximum(
            tl.load(totals_ptr + image * 2 + 1) / count - shifted_mean * shifted_mean,
            0.0,
        )
        std = tl.sqrt(var + eps)
        mean = shifted_mean + offset
        if block == 0:
            tl.store(std_ptr + image, std.to(std_ptr.dtype.element_ty))
        t = block * block_t + tl.arange(0, block_t)
        c = tl.arange(0, block_c)
        rows = t[:, None] < tokens
        real_c = c[None, :] < real
        x = tl.load(
            base + t[:, None].to(tl.int64) * padded + c[None, :], rows & real_c, 0
        )
        weight = tl.load(weight_ptr + c, c < real, 0).to(tl.float32)
        bias = tl.load(bias_ptr + c, c < real, 0).to(tl.float32)
        y = (x.to(tl.float32) - mean) / std * weight[None, :] + bias[None, :]
        y = tl.where(real_c, y, 0.0)
        tl.store(
            out_ptr
            + image.to(tl.int64) * tokens * padded
            + t[:, None].to(tl.int64) * padded
            + c[None, :],
            y.to(out_ptr.dtype.element_ty),
            rows & (c[None, :] < padded),
        )


def launch(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, real: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(normalized x, std (B, 1, 1)) for x (B, L, padded) contiguous."""
    b, tokens, padded = x.shape
    blocks = triton.cdiv(tokens, _TOKENS)
    block_c = max(16, 1 << (padded - 1).bit_length())
    sums = torch.empty((b, blocks, 2), device=x.device, dtype=torch.float32)
    out = torch.empty_like(x)
    std = torch.empty((b, 1, 1), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _iln_sums[(blocks, b)](
            x, sums, tokens, padded, real, blocks, block_t=_TOKENS, block_c=block_c
        )
        totals = sums.sum(1)
        _iln_apply[(blocks, b)](
            x,
            totals,
            weight,
            bias,
            out,
            std,
            tokens,
            padded,
            real,
            eps,
            block_t=_TOKENS,
            block_c=block_c,
        )
    return out, std


def usable(x: torch.Tensor, module: nn.Module) -> bool:
    """Whether the kernels may run: plain eager inference on an NVIDIA GPU, x FP16 or
    BF16 in the weights' type and contiguous."""
    return (
        ILN_KERNEL
        and x.is_cuda
        and torch.version.hip is None
        and x.dim() == 3
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.dtype == module.get_parameter("weight").dtype
        and x.is_contiguous()
        and plain_inference(x, module)
    )


def iln(
    module: nn.Module, x: torch.Tensor, real: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The kernels' i-LN of x, or None when the caller's PyTorch path must run: the
    kernels may not run (usable), or they failed in Triton, which switches them off for
    the rest of the process with one warning. Other errors are raised."""
    global ILN_KERNEL
    if not usable(x, module):
        return None
    try:
        return launch(
            x, module.get_parameter("weight"), module.get_parameter("bias"), real, eps
        )
    except Exception as error:
        if not from_triton(error):
            raise
        ILN_KERNEL = False
        logger.warning(
            "The Triton i-LN failed (%s: %s); using PyTorch's operations",
            type(error).__name__,
            error,
        )
        return None


class PaddedILN(nn.Module):
    """i-LN of `source`'s channels over the first of `padded` channels: statistics of the
    real channels, their weight and bias, zeros on the rest; returns (x, std) as i-LN does.
    In plain eager CUDA inference the kernels run; otherwise the same values from
    PyTorch's operations in FP32 over the real channels, returned in x's type."""

    def __init__(self, source: nn.Module, padded: int) -> None:
        super().__init__()
        real = source.get_parameter("weight").shape[0]
        if padded < real:
            raise ValueError("padded must be at least the norm's width")
        pad = (0, padded - real)
        self.weight = nn.Parameter(
            nn.functional.pad(source.get_parameter("weight").detach(), pad),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            nn.functional.pad(source.get_parameter("bias").detach(), pad),
            requires_grad=False,
        )
        self.eps = float(source.eps)
        self.real = real
        self.padded = padded

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = iln(self, x, self.real, self.eps)
        if result is not None:
            return result
        # the statistics of the real channels and the normalization in FP32, as the
        # kernels
        b = x.shape[0]
        wide = x.float()
        var, mean = torch.var_mean(
            wide[..., : self.real].reshape(b, -1), dim=1, keepdim=True, correction=0
        )
        std = torch.sqrt(var + self.eps).reshape(b, 1, 1)
        gain = self.weight.float() / std
        out = torch.addcmul(
            self.bias.float() - mean.reshape(b, 1, 1) * gain, wide, gain
        )
        return out.to(x.dtype), std.to(x.dtype)
