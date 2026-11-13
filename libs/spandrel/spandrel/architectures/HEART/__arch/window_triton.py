"""HEART's window attention as one Triton kernel for eager inference, reading q, k and v
where the qkv projection wrote them and writing each token's output where the output
projection reads it.

The kernel addresses the windows itself: a window's query and key rows are the image's
tokens at the window's place, past the half-window reflect padding of shifted blocks, so no
pass pads, partitions or merges the windows. The RIB position features, one (heads, N, R)
pair for every window, are a second dot product on the scores instead of columns
concatenated onto q and k. An online FP32 softmax over tiles of keys, as flash_window's.

Triton is optional: without it, AVAILABLE is False and HEART's PyTorch path runs.
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
HEART_WINDOW = AVAILABLE

if AVAILABLE:

    @triton.jit
    def _reflect(i, n):
        i = tl.where(i < 0, -i, i)
        return tl.where(i >= n, 2 * (n - 1) - i, i)

    @triton.jit
    def _heart_window(
        qkv_ptr,
        pos_q_ptr,
        pos_k_ptr,
        out_ptr,
        height: tl.int32,
        width: tl.int32,
        windows_w: tl.int32,
        windows_per_image: tl.int32,
        heads: tl.int32,
        shift: tl.int32,
        channels: tl.int32,
        head_width: tl.int32,
        rank: tl.int32,
        scale: tl.float32,
        window: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        block_r: tl.constexpr,
    ):
        """One program: block_m queries of one window and head against the window's
        window * window keys. qkv (B, H, W, 3 C) contiguous, q, k, v in that order and
        each C = heads * head_width; pos_q, pos_k (heads, N, rank) contiguous, pos_q
        already scaled; out (B, H, W, C) contiguous."""
        operand = out_ptr.dtype.element_ty
        qb, wh = tl.program_id(0), tl.program_id(1)
        win, head = wh // heads, wh % heads
        image = win // windows_per_image
        wy = (win % windows_per_image) // windows_w
        wx = win % windows_w
        n_tokens = window * window
        row_stride = 3 * channels

        m = qb * block_m + tl.arange(0, block_m)
        qy = wy * window + m // window - shift
        qx = wx * window + m % window - shift
        # 64-bit offsets: a large image's qkv holds more than 2**31 elements
        q_token = (
            (image * height + _reflect(qy, height)) * width + _reflect(qx, width)
        ).to(tl.int64)
        d = tl.arange(0, block_d)
        r = tl.arange(0, block_r)
        q = tl.load(
            qkv_ptr + q_token[:, None] * row_stride + head * head_width + d[None, :],
            d[None, :] < head_width,
            0,
        )
        pq = tl.load(
            pos_q_ptr + (head * n_tokens + m[:, None]) * rank + r[None, :],
            r[None, :] < rank,
            0,
        )
        qk_scale = scale * 1.4426950408889634
        maximum = tl.full((block_m,), -float("inf"), tl.float32)
        denominator = tl.full((block_m,), 0, tl.float32)
        acc = tl.full((block_m, block_d), 0, tl.float32)
        for start in range(0, n_tokens, block_n):
            n = start + tl.arange(0, block_n)
            ky = wy * window + n // window - shift
            kx = wx * window + n % window - shift
            k_token = (
                (image * height + _reflect(ky, height)) * width + _reflect(kx, width)
            ).to(tl.int64)
            k = tl.load(
                qkv_ptr
                + k_token[None, :] * row_stride
                + channels
                + head * head_width
                + d[:, None],
                d[:, None] < head_width,
                0,
            )
            v = tl.load(
                qkv_ptr
                + k_token[:, None] * row_stride
                + 2 * channels
                + head * head_width
                + d[None, :],
                d[None, :] < head_width,
                0,
            )
            pk = tl.load(
                pos_k_ptr + (head * n_tokens + n[None, :]) * rank + r[:, None],
                r[:, None] < rank,
                0,
            )
            score = tl.dot(q, k) * qk_scale + tl.dot(pq, pk) * 1.4426950408889634
            new_max = tl.maximum(maximum, tl.max(score, 1))
            p = tl.exp2(score - new_max[:, None])
            alpha = tl.exp2(maximum - new_max)
            acc = tl.dot(p.to(operand), v, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            maximum = new_max
        # the window's tokens inside the image, each written by exactly one window
        inside = (qy >= 0) & (qy < height) & (qx >= 0) & (qx < width)
        out_token = ((image * height + qy) * width + qx).to(tl.int64)
        tl.store(
            out_ptr + out_token[:, None] * channels + head * head_width + d[None, :],
            (acc / denominator[:, None]).to(operand),
            inside[:, None] & (d[None, :] < head_width),
        )


def _block(width: int) -> int:
    """A width's tile: the next power of two, at least 16 (tl.dot's minimum)."""
    return max(16, 1 << (width - 1).bit_length())


def launch(
    qkv: torch.Tensor,
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    heads: int,
    window: int,
    shift: int,
    scale: float,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """The attention's output (B, H, W, C) for qkv (B, H, W, 3 C) contiguous, H and W
    multiples of the window, and pos_q, pos_k (heads, N, R) contiguous."""
    b, h, w, c3 = qkv.shape
    c = c3 // 3
    rank = pos_q.shape[-1]
    windows_h, windows_w = (h + 2 * shift) // window, (w + 2 * shift) // window
    out = qkv.new_empty((b, h, w, c))
    grid = (
        triton.cdiv(window * window, block_m),
        b * windows_h * windows_w * heads,
    )
    with torch.cuda.device(qkv.device):
        _heart_window[grid](
            qkv,
            pos_q,
            pos_k,
            out,
            h,
            w,
            windows_w,
            windows_h * windows_w,
            heads,
            shift,
            c,
            c // heads,
            rank,
            scale,
            window=window,
            block_m=block_m,
            block_n=block_n,
            block_d=_block(c // heads),
            block_r=_block(rank),
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


_WEIGHTLESS = nn.Module().eval()


def usable(qkv: torch.Tensor, window: int) -> bool:
    """Whether the kernel may run: plain eager inference on an NVIDIA GPU, FP16 or BF16,
    a contiguous qkv, and windows of a whole number of key tiles."""
    return (
        HEART_WINDOW
        and qkv.is_cuda
        and torch.version.hip is None
        and qkv.dtype in (torch.float16, torch.bfloat16)
        and qkv.is_contiguous()
        and (window * window) % 64 == 0
        and plain_inference(qkv, _WEIGHTLESS)
    )


def heart_window_attention(
    qkv: torch.Tensor,
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    heads: int,
    window: int,
    shift: int,
    scale: float,
) -> torch.Tensor | None:
    """The kernel's output, or None when the caller's PyTorch path must run: the kernel
    may not run (usable), or it failed in Triton, which switches it off for the rest of
    the process with one warning. Other errors are raised."""
    global HEART_WINDOW
    if not usable(qkv, window):
        return None
    try:
        return launch(qkv, pos_q, pos_k, heads, window, shift, scale)
    except Exception as error:
        if not from_triton(error):
            raise
        HEART_WINDOW = False
        logger.warning(
            "The Triton HEART window attention failed (%s: %s); using PyTorch's attention",
            type(error).__name__,
            error,
        )
        return None
