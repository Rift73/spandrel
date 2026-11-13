"""DUAL's local and retrieval neighborhood attention as Triton kernels for eager
inference, adapted from the TensorRT plugin kernels of chaiNNer-C's DUAL NA study
(8 x 8 query blocks, 64-key blocks, online FP32 softmax): each program reads its
queries' neighborhoods straight from the projections' outputs and writes the attention
output once; no gathered keys, masks or chunks.

Local (_SpatialAttention._local_na_core): query (y, x) of an H x W map attends the
keys start = clamp(y - K // 2, 0, H - K) + [0, K), K = min(window, H) per axis, with
the dense relative position bias of the query-minus-key offset. Q is scaled by
1/sqrt(32) and rounded to the operands' type, and the bias is rounded to it, as the
PyTorch path rounds them.

Retrieval (attend_packed with NAGeometry): fine query (y, x) attends the coarse keys
start = clamp(y // 2 - K // 2, 0, Hc - K) + [0, K), K = min(48, Hc) per axis, no
bias; Q is scaled by the head's tau (rounded to the operands' type) and the logits by
log(Ky * Kx) / (log(1024) * sqrt(32)).

The image-sized count is on the launch grid's first axis (CUDA caps the others at
65,535 programs). Triton is optional: without it, AVAILABLE is False and the PyTorch
path runs.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    AVAILABLE = False
else:
    AVAILABLE = True

# 8 x 8 queries per program, 64 keys per block (the study's best TensorRT shape)
QH = QW = 8
BLOCK_N = 64

if AVAILABLE:

    @triton.jit
    def _local_na(
        qkv_ptr,
        table_ptr,
        out_ptr,
        height: tl.int32,
        width: tl.int32,
        tiles: tl.int32,
        channels: tl.constexpr,
        window: tl.constexpr,
        qh: tl.constexpr,
        qw: tl.constexpr,
        block_n: tl.constexpr,
    ):
        """qkv [B, H, W, 3C] (all Q, all K, all V; heads of 32), table [C / 32,
        (2 window - 1)^2] and out [B, H, W, C], one 16-bit type."""
        operand = out_ptr.dtype.element_ty
        block_m: tl.constexpr = qh * qw
        program, head = tl.program_id(0), tl.program_id(1)
        image, tile = program // tiles, program % tiles
        base = image.to(tl.int64) * height * width
        oy = tile // tl.cdiv(width, qw) * qh
        ox = tile % tl.cdiv(width, qw) * qw
        m = tl.arange(0, block_m)
        qy = tl.minimum(oy + m // qw, height - 1)
        qx = tl.minimum(ox + m % qw, width - 1)
        us: tl.constexpr = window + qw - 1
        stride: tl.constexpr = 2 * window - 1
        half: tl.constexpr = window // 2
        kh, kw = tl.minimum(height, window), tl.minimum(width, window)
        rows = tl.minimum(height, kh + qh - 1)
        cols = tl.minimum(width, kw + qw - 1)
        ky0 = tl.minimum(
            tl.maximum(tl.minimum(oy, height - 1) - half, 0), height - rows
        )
        kx0 = tl.minimum(tl.maximum(tl.minimum(ox, width - 1) - half, 0), width - cols)
        d = tl.arange(0, 32)
        n = tl.arange(0, block_n)
        pix = base + qy * width + qx
        q = tl.load(qkv_ptr + pix[:, None] * (3 * channels) + head * 32 + d[None, :])
        q = (q.to(tl.float32) * 0.1767766952966369).to(operand)
        sy = tl.minimum(tl.maximum(qy - half, 0), height - kh)
        sx = tl.minimum(tl.maximum(qx - half, 0), width - kw)
        maximum = tl.full((block_m,), -float("inf"), tl.float32)
        denominator = tl.full((block_m,), 0, tl.float32)
        acc = tl.full((block_m, 32), 0, tl.float32)
        trips: tl.constexpr = ((window + qh - 1) * us + block_n - 1) // block_n
        for start in range(0, trips * block_n, block_n):
            ns = start + n
            ky, kx = ky0 + ns // us, kx0 + ns % us
            inside = (ns < rows * us) & (ns % us < cols)
            kp = base + ky * width + kx
            k = tl.load(
                qkv_ptr
                + kp[None, :] * (3 * channels)
                + channels
                + head * 32
                + d[:, None],
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
            valid = (
                (ky[None, :] >= sy[:, None])
                & (ky[None, :] < sy[:, None] + kh)
                & (kx[None, :] >= sx[:, None])
                & (kx[None, :] < sx[:, None] + kw)
            )
            dy = tl.minimum(
                tl.maximum(qy[:, None] - ky[None, :] + window - 1, 0), stride - 1
            )
            dx = tl.minimum(
                tl.maximum(qx[:, None] - kx[None, :] + window - 1, 0), stride - 1
            )
            bias = tl.load(table_ptr + head * stride * stride + dy * stride + dx).to(
                tl.float32
            )
            score = (tl.dot(q, k) + bias) * 1.4426950408889634
            score = tl.where(valid, score, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            # A query may have no valid key yet: avoid -inf - (-inf) until its first one.
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            p = tl.exp2(score - safe_max[:, None])
            alpha = tl.exp2(maximum - safe_max)
            acc = tl.dot(p.to(operand), v, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            maximum = new_max
        # Lanes past the map's edge repeat the last valid query and are not stored.
        stored = (oy + m // qw < height) & (ox + m % qw < width)
        tl.store(
            out_ptr + pix[:, None] * channels + head * 32 + d[None, :],
            (acc / denominator[:, None]).to(operand),
            stored[:, None],
        )

    @triton.jit
    def _retrieval_na(
        q_ptr,
        k_ptr,
        v_ptr,
        tau_ptr,
        out_ptr,
        height: tl.int32,
        width: tl.int32,
        kheight: tl.int32,
        kwidth: tl.int32,
        tiles: tl.int32,
        scale: tl.float32,
        heads: tl.constexpr,
        neighbourhood: tl.constexpr,
        qh: tl.constexpr,
        qw: tl.constexpr,
        block_n: tl.constexpr,
    ):
        """q and out [B, C, H, W], k and v [B, C, Hc, Wc] (heads of 32), tau [heads],
        all contiguous in one 16-bit type."""
        operand = out_ptr.dtype.element_ty
        block_m: tl.constexpr = qh * qw
        program, head = tl.program_id(0), tl.program_id(1)
        image, tile = program // tiles, program % tiles
        q_plane = height * width
        k_plane = kheight * kwidth
        q_base = (image.to(tl.int64) * heads + head) * 32 * q_plane
        k_base = (image.to(tl.int64) * heads + head) * 32 * k_plane
        oy = tile // tl.cdiv(width, qw) * qh
        ox = tile % tl.cdiv(width, qw) * qw
        m = tl.arange(0, block_m)
        qy = tl.minimum(oy + m // qw, height - 1)
        qx = tl.minimum(ox + m % qw, width - 1)
        cy, cx = qy // 2, qx // 2
        us: tl.constexpr = neighbourhood + qw // 2 - 1
        kh, kw = tl.minimum(kheight, neighbourhood), tl.minimum(kwidth, neighbourhood)
        rows = tl.minimum(kheight, kh + qh // 2 - 1)
        cols = tl.minimum(kwidth, kw + qw // 2 - 1)
        half: tl.constexpr = neighbourhood // 2
        ky0 = tl.minimum(
            tl.maximum(tl.minimum(oy, height - 1) // 2 - half, 0), kheight - rows
        )
        kx0 = tl.minimum(
            tl.maximum(tl.minimum(ox, width - 1) // 2 - half, 0), kwidth - cols
        )
        d = tl.arange(0, 32)
        n = tl.arange(0, block_n)
        pix = qy * width + qx
        tau = tl.load(tau_ptr + head)
        q = tl.load(q_ptr + q_base + d[None, :] * q_plane + pix[:, None])
        q = (q.to(tl.float32) * tau.to(tl.float32)).to(operand)
        sy = tl.minimum(tl.maximum(cy - half, 0), kheight - kh)
        sx = tl.minimum(tl.maximum(cx - half, 0), kwidth - kw)
        maximum = tl.full((block_m,), -float("inf"), tl.float32)
        denominator = tl.full((block_m,), 0, tl.float32)
        acc = tl.full((block_m, 32), 0, tl.float32)
        trips: tl.constexpr = (
            (neighbourhood + qh // 2 - 1) * us + block_n - 1
        ) // block_n
        for start in range(0, trips * block_n, block_n):
            ns = start + n
            ky, kx = ky0 + ns // us, kx0 + ns % us
            inside = (ns < rows * us) & (ns % us < cols)
            kp = ky * kwidth + kx
            k = tl.load(
                k_ptr + k_base + d[:, None] * k_plane + kp[None, :], inside[None, :], 0
            )
            v = tl.load(
                v_ptr + k_base + d[None, :] * k_plane + kp[:, None], inside[:, None], 0
            )
            score = tl.dot(q, k) * (scale * 1.4426950408889634)
            valid = (
                (ky[None, :] >= sy[:, None])
                & (ky[None, :] < sy[:, None] + kh)
                & (kx[None, :] >= sx[:, None])
                & (kx[None, :] < sx[:, None] + kw)
            )
            score = tl.where(valid, score, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            p = tl.exp2(score - safe_max[:, None])
            alpha = tl.exp2(maximum - safe_max)
            acc = tl.dot(p.to(operand), v, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            maximum = new_max
        stored = (oy + m // qw < height) & (ox + m % qw < width)
        tl.store(
            out_ptr + q_base + d[None, :] * q_plane + pix[:, None],
            (acc / denominator[:, None]).to(operand),
            stored[:, None],
        )


def local_na(
    qkv: torch.Tensor, table: torch.Tensor, height: int, width: int, window: int
) -> torch.Tensor:
    """qkv B,H*W,3C (unscaled Q, K, V; heads of 32) and the bias table heads,
    (2 window - 1)^2, contiguous in one 16-bit type, to the attention output B,H,W,C."""
    batch, channels = qkv.shape[0], qkv.shape[-1] // 3
    out = qkv.new_empty((batch, height, width, channels))
    tiles = triton.cdiv(height, QH) * triton.cdiv(width, QW)
    with torch.cuda.device(qkv.device):
        _local_na[(batch * tiles, channels // 32)](
            qkv,
            table,
            out,
            height,
            width,
            tiles,
            channels=channels,
            window=window,
            qh=QH,
            qw=QW,
            block_n=BLOCK_N,
            num_warps=4,
            num_stages=1,
        )
    return out


def retrieval_na(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    window: int = 48,
) -> torch.Tensor:
    """q B,C,H,W and k, v B,C,Hc,Wc (heads of 32), tau heads, contiguous in one 16-bit
    type, to the attention output B,C,H,W."""
    batch, channels, height, width = q.shape
    kheight, kwidth = k.shape[-2:]
    keys = min(window, kheight) * min(window, kwidth)
    scale = math.log(keys) / (math.log(1024) * math.sqrt(32))
    out = torch.empty_like(q)
    tiles = triton.cdiv(height, QH) * triton.cdiv(width, QW)
    with torch.cuda.device(q.device):
        _retrieval_na[(batch * tiles, channels // 32)](
            q,
            k,
            v,
            tau,
            out,
            height,
            width,
            kheight,
            kwidth,
            tiles,
            scale,
            heads=channels // 32,
            neighbourhood=window,
            qh=QH,
            qw=QW,
            block_n=BLOCK_N,
            num_warps=4,
            num_stages=1,
        )
    return out
