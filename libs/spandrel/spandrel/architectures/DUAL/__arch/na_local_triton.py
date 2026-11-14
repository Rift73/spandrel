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

Regional (RegionalChannelAttention._mix_na, regional_na): position (y, x) and head h
take the 32 x 32 channel attention of the clamped Ky x Kx neighborhood, K = min(32, L)
per axis: Gram and lengths (+ Ky * Kx * 1e-6) in FP32 from the FP16 q and k, cosine *
tau, FP32 softmax, rounded to the operands' type and applied to the position's own v.
The Gram slides: a program takes one column and a segment of SEGMENT rows for one
head; it sums the first window from its row strips (32 x 32 x 32 tl.dot each), then
per row adds the entering strip and subtracts the leaving one when the clamped window
moves. The running sums restart exactly at every segment, which bounds their drift,
and at any row where a square sum fell below 1 / RESTART of the largest it held in
the segment: the rounding a large value left behind cannot swamp a small window.

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
        # 64-bit: a channel's plane offset d * plane passes 2^31 on large maps
        d = tl.arange(0, 32).to(tl.int64)
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

    @triton.jit
    def _strip(ptr, channel, row, col0, col_stride, live, channels_first: tl.constexpr):
        """One neighborhood row strip (row: its 64-bit offset): [32 d, 32 columns] when
        channels_first, else [32 columns, 32 d] (zeros where not live)."""
        pix = row + (col0 + tl.arange(0, 32)).to(tl.int64) * col_stride
        if channels_first:
            return tl.load(ptr + channel[:, None] + pix[None, :], live[None, :], 0)
        return tl.load(ptr + channel[None, :] + pix[:, None], live[:, None], 0)

    @triton.jit
    def _window_gram(
        q_ptr, k_ptr, channel, row0, kh, row_stride, col0, col_stride, columns
    ):
        """The Gram and the square sums of the window whose rows start at row0,
        summed from its row strips."""
        gram = tl.zeros((32, 32), tl.float32)
        q_square = tl.zeros((32,), tl.float32)
        k_square = tl.zeros((32,), tl.float32)
        for r in range(32):
            live = columns & (r < kh)
            row = (row0 + r).to(tl.int64) * row_stride
            qt = _strip(q_ptr, channel, row, col0, col_stride, live, True)
            k = _strip(k_ptr, channel, row, col0, col_stride, live, False)
            gram = tl.dot(qt, k, gram)
            qf, kf = qt.to(tl.float32), k.to(tl.float32)
            q_square += tl.sum(qf * qf, 1)
            k_square += tl.sum(kf * kf, 0)
        return gram, q_square, k_square

    @triton.jit
    def _regional_na(
        q_ptr,
        k_ptr,
        v_ptr,
        tau_ptr,
        out_ptr,
        height: tl.int32,
        width: tl.int32,
        segments: tl.int32,
        image_stride,
        channel_stride,
        row_stride,
        col_stride,
        out_image_stride,
        out_channel_stride,
        out_row_stride,
        out_col_stride,
        segment: tl.constexpr,
        restart: tl.constexpr,
    ):
        """q, k and v [B, C, H, W] with one set of strides (views of one depthwise
        output in either memory format), tau [C / 32] FP32, out [B, C, H, W]; heads of
        32, one 16-bit type."""
        operand = out_ptr.dtype.element_ty
        program, head = tl.program_id(0), tl.program_id(1)
        image = (program // (segments * width)).to(tl.int64)
        x = program % width
        y0 = program // width % segments * segment
        kh, kw = tl.minimum(height, 32), tl.minimum(width, 32)
        col0 = tl.minimum(tl.maximum(x - 16, 0), width - kw)
        row0 = tl.minimum(tl.maximum(y0 - 16, 0), height - kh)
        # 64-bit: offsets pass 2^31 on large maps
        d = (head * 32 + tl.arange(0, 32)).to(tl.int64)
        channel = image * image_stride + d * channel_stride
        columns = tl.arange(0, 32) < kw
        count = (kh * kw).to(tl.float32) * 1e-6
        scale = tl.load(tau_ptr + head) * 1.4426950408889634
        gram, q_square, k_square = _window_gram(
            q_ptr, k_ptr, channel, row0, kh, row_stride, col0, col_stride, columns
        )
        q_peak, k_peak = q_square, k_square
        out = out_ptr + image * out_image_stride + d * out_channel_stride
        for i in range(segment):
            y = y0 + i
            moved = (tl.minimum(tl.maximum(y - 16, 0), height - kh) > row0) & (
                y < height
            )
            live = columns & moved
            entering = (row0 + kh).to(tl.int64) * row_stride
            leaving = row0.to(tl.int64) * row_stride
            qa = _strip(q_ptr, channel, entering, col0, col_stride, live, True)
            ka = _strip(k_ptr, channel, entering, col0, col_stride, live, False)
            ql = _strip(q_ptr, channel, leaving, col0, col_stride, live, True)
            kl = _strip(k_ptr, channel, leaving, col0, col_stride, live, False)
            gram = tl.dot(qa, ka, gram) - tl.dot(ql, kl)
            qaf, kaf = qa.to(tl.float32), ka.to(tl.float32)
            qlf, klf = ql.to(tl.float32), kl.to(tl.float32)
            q_square += tl.sum(qaf * qaf, 1)
            k_square += tl.sum(kaf * kaf, 0)
            q_peak = tl.maximum(q_peak, q_square)
            k_peak = tl.maximum(k_peak, k_square)
            q_square -= tl.sum(qlf * qlf, 1)
            k_square -= tl.sum(klf * klf, 0)
            row0 = tl.where(moved, row0 + 1, row0)
            # The running sums carry rounding of the largest sums they held; once
            # those are RESTART times today's, the window is summed from scratch.
            # One reduction per vector (q and k sums sit in different layouts; an
            # elementwise maximum of the two would convert one of them every row).
            # The test is negated so that a non-finite sum restarts too: once an inf
            # or NaN has left the window, its rows are finite again.
            q_held = tl.min((q_peak <= restart * (q_square + count)).to(tl.int32))
            k_held = tl.min((k_peak <= restart * (k_square + count)).to(tl.int32))
            if q_held * k_held == 0:
                gram, q_square, k_square = _window_gram(
                    q_ptr,
                    k_ptr,
                    channel,
                    row0,
                    kh,
                    row_stride,
                    col0,
                    col_stride,
                    columns,
                )
                q_peak, k_peak = q_square, k_square
            q_norm = tl.sqrt(q_square + count)
            k_norm = tl.sqrt(k_square + count)
            logits = gram / (q_norm[:, None] * k_norm[None, :]) * scale
            p = tl.exp2(logits - tl.max(logits, 1)[:, None])
            a = (p / tl.sum(p, 1)[:, None]).to(operand).to(tl.float32)
            yc = tl.minimum(y, height - 1).to(tl.int64)
            # x stays 32-bit (an int64 x costs registers through col0 and the
            # strips); its two address products are 64-bit
            x_wide = x.to(tl.int64)
            v = tl.load(v_ptr + channel + yc * row_stride + x_wide * col_stride)
            tl.store(
                out + yc * out_row_stride + x_wide * out_col_stride,
                tl.sum(a * v.to(tl.float32)[None, :], 1).to(operand),
                y < height,
            )


# rows per regional program: the running Gram restarts exactly every SEGMENT rows,
# and within a segment wherever a square sum fell below 1 / RESTART of its peak
SEGMENT, RESTART = 32, 64.0
# Registers per thread of the one-warp regional programs: at 168 twelve programs fit
# an SM's 65,536 registers; the restart path lets the compiler take 218, and from 176
# up the kernel is 9-15 % slower (2x DUAL Light, 960x540, RTX 5090). At 168 it spills
# a few registers and runs at the speed it had before the restart test.
REGIONAL_REGISTERS = 168


def regional_na(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """q, k and v B,C,H,W (heads of 32; one set of strides, one 16-bit type) and tau
    heads (FP32) to the regional attention output B,C,H,W, channels-last when q is."""
    batch, channels, height, width = q.shape
    layout = torch.channels_last if q.stride(1) == 1 else torch.contiguous_format
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device, memory_format=layout)
    segments = triton.cdiv(height, SEGMENT)
    with torch.cuda.device(q.device):
        _regional_na[(batch * segments * width, channels // 32)](
            q,
            k,
            v,
            tau,
            out,
            height,
            width,
            segments,
            *q.stride(),
            *out.stride(),
            segment=SEGMENT,
            restart=RESTART,
            num_warps=1,
            num_stages=1,
            maxnreg=REGIONAL_REGISTERS,
        )
    return out


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
