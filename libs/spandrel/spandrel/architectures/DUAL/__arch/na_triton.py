"""DUAL's neighborhood attention (NA) core as one Triton kernel for eager inference.

The kernel of chaiNNer-C's DualNA TensorRT plugin (its BF16 casts follow the tensors'
own type here, so FP16 runs too): exact inward-clamped neighborhoods with the learned
relative Fourier bias (RFB) given as two factor tables, an online FP32 softmax and the
value product, fused. It reads Q, K and V straight from the projection's output and
writes the attention output once: no gathered keys, masks or chunks.

For a query (y, x) of an H x W trunk and window M (32 or 64):
    Kh = min(H, M), sy = clamp(y - Kh // 2, 0, H - Kh), key rows [sy, sy + Kh)
(columns alike with W), the operator of _SpatialAttention._na_core. Q is scaled by
1/sqrt(32) and rounded to the operands' type, as the PyTorch path rounds it. The RFB
factors are indexed in the union-local frame of 16 x 16 query tiles that the PyTorch
path uses (row stride M + 15), so the bias terms are the same values.

Triton is optional: without it, AVAILABLE is False and the PyTorch path runs.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    AVAILABLE = False
else:
    AVAILABLE = True

if AVAILABLE:

    @triton.jit
    def _na_attention(
        qkv_ptr,
        bq_ptr,
        bk_ptr,
        height: tl.int32,
        width: tl.int32,
        out_ptr,
        channels: tl.constexpr,
        window: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        """qkv [H, W, 3C] (all Q, all K, all V; heads of 32), factor tables
        [C / 32, (M + 15)^2, 16] and out [H, W, C], all in one 16-bit type."""
        tl.static_assert(block_m == 64)
        tl.static_assert(block_n == 64)
        operand = out_ptr.dtype.element_ty
        tile, sub, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        ty = tile // tl.cdiv(width, 16) * 16
        tx = tile % tl.cdiv(width, 16) * 16
        m = sub * block_m + tl.arange(0, block_m)
        qy = tl.minimum(ty + m // 16, height - 1)
        qx = tl.minimum(tx + m % 16, width - 1)
        u: tl.constexpr = window + 15
        kh, kw = tl.minimum(height, window), tl.minimum(width, window)
        uh, uw = tl.minimum(height, kh + 15), tl.minimum(width, kw + 15)
        by = tl.minimum(tl.maximum(ty - kh // 2, 0), height - uh)
        bx = tl.minimum(tl.maximum(tx - kw // 2, 0), width - uw)
        # A program's four query rows need at most window + 3 key rows: the 64-key
        # blocks start from those rows' first key row, on the 128-query parent's grid.
        rows = tl.minimum(height, kh + 7)
        ky0 = tl.minimum(tl.maximum(ty + (sub // 2) * 8 - kh // 2, 0), height - rows)
        d = tl.arange(0, 32)
        r = tl.arange(0, 16)
        n = tl.arange(0, block_n)
        pix = qy * width + qx
        q = tl.load(qkv_ptr + pix[:, None] * (3 * channels) + head * 32 + d[None, :])
        q = (q.to(tl.float32) * 0.1767766952966369).to(operand)
        bqi = (qy - by) * u + (qx - bx)
        bq = tl.load(bq_ptr + (head * u * u + bqi[:, None]) * 16 + r[None, :])
        sy = tl.minimum(tl.maximum(qy - kh // 2, 0), height - kh)
        sx = tl.minimum(tl.maximum(qx - kw // 2, 0), width - kw)
        maximum = tl.full((block_m,), -float("inf"), tl.float32)
        denominator = tl.full((block_m,), 0, tl.float32)
        acc = tl.full((block_m, 32), 0, tl.float32)
        first_qy = tl.minimum(ty + sub * 4, height - 1)
        first_sy = tl.minimum(tl.maximum(first_qy - kh // 2, 0), height - kh)
        first = ((first_sy - ky0) * u // block_n) * block_n
        trips: tl.constexpr = ((window + 3) * u + 2 * block_n - 2) // block_n
        for start in range(0, trips * block_n, block_n):
            ns = first + start + n
            ky, kx = ky0 + ns // u, bx + ns % u
            # The union's row stride stays M + 15; columns past its width are not
            # loaded, and keys outside it fail every query's neighborhood test.
            inside = (ns < rows * u) & (ns % u < uw)
            kp = ky * width + kx
            bi = (ky - by) * u + kx - bx
            k = tl.load(
                qkv_ptr
                + kp[None, :] * (3 * channels)
                + channels
                + head * 32
                + d[:, None],
                inside[None, :],
                0,
            )
            bk = tl.load(
                bk_ptr + (head * u * u + bi[None, :]) * 16 + r[:, None],
                inside[None, :],
                0,
            )
            v = tl.load(
                qkv_ptr
                + kp[:, None] * (3 * channels)
                + 2 * channels
                + head * 32
                + d[None, :],
                inside[:, None],
                0,
            )
            score = tl.dot(q, k)
            score = tl.dot(bq, bk, score) * 1.4426950408889634
            valid = (
                (ky[None, :] >= sy[:, None])
                & (ky[None, :] < sy[:, None] + kh)
                & (kx[None, :] >= sx[:, None])
                & (kx[None, :] < sx[:, None] + kw)
            )
            score = tl.where(valid, score, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            # A query may have no valid key yet: avoid -inf - (-inf) until its first one.
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            p = tl.exp2(score - safe_max[:, None])
            alpha = tl.exp2(maximum - safe_max)
            acc = tl.dot(p.to(operand), v, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            maximum = new_max
        # Lanes past the trunk's edge repeat the last valid query and are not stored.
        stored = (ty + m // 16 < height) & (tx + m % 16 < width)
        tl.store(
            out_ptr + pix[:, None] * channels + head * 32 + d[None, :],
            (acc / denominator[:, None]).to(operand),
            stored[:, None],
        )


def na_attention(
    qkv: torch.Tensor,
    query_factors: torch.Tensor,
    key_factors: torch.Tensor,
    height: int,
    width: int,
    window: int,
) -> torch.Tensor:
    """qkv B,H*W,3C (unscaled Q, K, V; heads of 32), factor tables heads,(M+15)^2,16,
    all contiguous in one 16-bit type, to the attention output B,H,W,C."""
    batch, channels = qkv.shape[0], qkv.shape[-1] // 3
    out = qkv.new_empty((batch, height, width, channels))
    grid = (triton.cdiv(height, 16) * triton.cdiv(width, 16), 4, channels // 32)
    with torch.cuda.device(qkv.device):
        for b in range(batch):
            _na_attention[grid](
                qkv[b],
                query_factors,
                key_factors,
                height,
                width,
                out[b],
                channels=channels,
                window=window,
                block_m=64,
                block_n=64,
                num_warps=4,
                num_stages=1,
            )
    return out
