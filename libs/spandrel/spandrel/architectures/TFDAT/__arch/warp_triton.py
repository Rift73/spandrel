"""TFDATv2's warp_frame as one Triton kernel for eager inference: border-replicated
bilinear sampling at an integer base position plus the flow's floor, weighted by the
flow's fraction, with each of the source's operations rounded to the input's precision
as PyTorch rounds it, so the result is the source's bit for bit. The PyTorch path makes
about 50 launches (and int64 index maps) per call, 24 calls per 9-frame clip: 0.5 to
0.7 ms per call at 720x480 on an RTX 5090, the kernel 0.02 to 0.07 ms.

Triton is optional: without it, AVAILABLE is False and the caller's PyTorch path runs.
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

logger = logging.getLogger("spandrel.attention")

# A kernel that fails to compile or launch turns this off for the rest of the process.
WARP = AVAILABLE

if AVAILABLE:

    @triton.jit
    def _warp(
        x_ptr,
        flow_ptr,
        out_ptr,
        channels,
        height,
        width,
        sn,
        sc,
        sy,
        sx,
        fn,
        fc,
        fy_,
        fx_,
        block: tl.constexpr,
    ):
        """One program: block pixels of one image, every channel. x and out share the
        strides (sn, sc, sy, sx); flow's are (fn, fc, fy_, fx_)."""
        n = tl.program_id(0).to(tl.int64)
        p = tl.program_id(1) * block + tl.arange(0, block)
        ok = p < height * width
        yy = p // width
        xx = p % width
        t = x_ptr.dtype.element_ty
        flow = flow_ptr + n * fn + yy * fy_ + xx * fx_
        fx = tl.load(flow, ok, 0.0).to(tl.float32)
        fy = tl.load(flow + fc, ok, 0.0).to(tl.float32)
        fx0 = tl.floor(fx)
        fy0 = tl.floor(fy)
        # each value as the source's operation in the input's precision leaves it
        dx = (fx - fx0).to(t).to(tl.float32)
        dy = (fy - fy0).to(t).to(tl.float32)
        x0 = xx + fx0.to(tl.int32)
        y0 = yy + fy0.to(tl.int32)
        x1 = tl.minimum(tl.maximum(x0 + 1, 0), width - 1)
        y1 = tl.minimum(tl.maximum(y0 + 1, 0), height - 1)
        x0 = tl.minimum(tl.maximum(x0, 0), width - 1)
        y0 = tl.minimum(tl.maximum(y0, 0), height - 1)
        ex = (1 - dx).to(t).to(tl.float32)
        ey = (1 - dy).to(t).to(tl.float32)
        w00 = (ex * ey).to(t).to(tl.float32)
        w01 = (dx * ey).to(t).to(tl.float32)
        w10 = (ex * dy).to(t).to(tl.float32)
        w11 = (dx * dy).to(t).to(tl.float32)
        for c in range(channels):
            plane = x_ptr + n * sn + c * sc
            p00 = tl.load(plane + y0 * sy + x0 * sx, ok, 0.0).to(tl.float32)
            p01 = tl.load(plane + y0 * sy + x1 * sx, ok, 0.0).to(tl.float32)
            p10 = tl.load(plane + y1 * sy + x0 * sx, ok, 0.0).to(tl.float32)
            p11 = tl.load(plane + y1 * sy + x1 * sx, ok, 0.0).to(tl.float32)
            v = (w00 * p00).to(t).to(tl.float32) + (w01 * p01).to(t).to(tl.float32)
            v = v.to(t).to(tl.float32) + (w10 * p10).to(t).to(tl.float32)
            v = v.to(t).to(tl.float32) + (w11 * p11).to(t).to(tl.float32)
            tl.store(out_ptr + n * sn + c * sc + yy * sy + xx * sx, v.to(t), ok)


# plain_inference's module: the warp has no weights.
_WEIGHTLESS = nn.Module().eval()


def usable(x: torch.Tensor, flow: torch.Tensor) -> bool:
    """Whether the kernel may warp x by flow: plain eager inference (plain_inference) on
    an NVIDIA GPU, FP16 or BF16 x and flow of one type, x in NCHW or channels-last order."""
    return (
        WARP
        and x.is_cuda
        and torch.version.hip is None
        and x.dtype in (torch.float16, torch.bfloat16)
        and flow.dtype == x.dtype
        and (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last))
        and plain_inference(x, _WEIGHTLESS)
        and plain_inference(flow, _WEIGHTLESS)
    )


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor | None:
    """warp_frame(x, flow) from the kernel, or None when the caller's PyTorch path must
    run: the kernel may not run (usable), or it failed in Triton, which switches it off
    for the rest of the process with one warning. Other errors are raised."""
    global WARP
    if not usable(x, flow):
        return None
    n, c, h, w = x.shape
    out = torch.empty_like(x)  # x's strides
    block = 256
    try:
        with torch.cuda.device(x.device):
            _warp[(n, triton.cdiv(h * w, block))](
                x, flow, out, c, h, w, *x.stride(), *flow.stride(), block=block
            )
    except Exception as error:
        if not from_triton(error):
            raise
        WARP = False
        logger.warning(
            "The Triton warp failed (%s: %s); using PyTorch's gathers",
            type(error).__name__,
            error,
        )
        return None
    return out
