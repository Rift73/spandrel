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

Routing (routing_na): the router's means over the windows at the distinct clamped
starts, as two passes of sliding window sums (rows, then columns) with FP64 running
sums; and the base part plus the coefficient-weighted expert parts of the wide
convolution's output, summed in FP32 in one pass and written in the parts' type.

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
        x = (program % width).to(tl.int64)
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
            excess = tl.maximum(
                q_peak - restart * (q_square + count),
                k_peak - restart * (k_square + count),
            )
            if tl.max(excess) > 0:
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
            v = tl.load(v_ptr + channel + yc * row_stride + x * col_stride)
            tl.store(
                out + yc * out_row_stride + x * out_col_stride,
                tl.sum(a * v.to(tl.float32)[None, :], 1).to(operand),
                y < height,
            )

    @triton.jit
    def _routed_sum(
        parts_ptr,
        coefficients_ptr,
        out_ptr,
        positions: tl.int32,
        channels: tl.int32,
        part_image,
        part_channel,
        part_position,
        coefficient_image,
        coefficient_expert,
        coefficient_position,
        out_image,
        out_channel,
        out_position,
        experts: tl.constexpr,
        block_p: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """parts [B, (K + 1) C, P] (the base, then the experts), coefficients [B, K, P]
        FP32 and out [B, C, P], P = H * W in either memory format."""
        # 64-bit: an expert part's offset passes 2^31 on large maps
        p = tl.program_id(0).to(tl.int64) * block_p + tl.arange(0, block_p)
        c = tl.program_id(1).to(tl.int64) * block_c + tl.arange(0, block_c)
        image = tl.program_id(2).to(tl.int64)
        part_stride = channels.to(tl.int64) * part_channel
        live = p < positions
        both = live[:, None] & (c < channels)[None, :]
        part = (
            parts_ptr
            + image * part_image
            + p[:, None] * part_position
            + c[None, :] * part_channel
        )
        acc = tl.load(part, both, 0).to(tl.float32)
        coefficient = (
            coefficients_ptr + image * coefficient_image + p * coefficient_position
        )
        for e in tl.static_range(experts):
            weight = tl.load(
                coefficient + tl.full((), e, tl.int64) * coefficient_expert, live, 0
            )
            value = tl.load(part + (e + 1) * part_stride, both, 0)
            acc += weight[:, None] * value.to(tl.float32)
        tl.store(
            out_ptr
            + image * out_image
            + p[:, None] * out_position
            + c[None, :] * out_channel,
            acc.to(out_ptr.dtype.element_ty),
            both,
        )

    @triton.jit
    def _window_sums(
        in_ptr,
        out_ptr,
        steps: tl.int32,
        lanes: tl.int32,
        channels: tl.int32,
        window: tl.int32,
        scale: tl.float32,
        in_image,
        in_channel,
        in_lane,
        in_step,
        out_image,
        out_channel,
        out_lane,
        out_step,
        block: tl.constexpr,
    ):
        """Sums of `window` consecutive steps for every lane of one (image, channel)
        plane, at the steps - window + 1 starts, times scale: an FP64 running sum adds
        the entering step and subtracts the leaving one."""
        # one axis: (plane, lane block) pairs, so B * C may pass 65,535
        blocks = tl.cdiv(lanes, block)
        plane = tl.program_id(0) // blocks
        lane = (tl.program_id(0) % blocks).to(tl.int64) * block + tl.arange(0, block)
        image, channel = (
            (plane // channels).to(tl.int64),
            (plane % channels).to(tl.int64),
        )
        live = lane < lanes
        source = in_ptr + image * in_image + channel * in_channel + lane * in_lane
        target = out_ptr + image * out_image + channel * out_channel + lane * out_lane
        running = tl.zeros((block,), tl.float64)
        for step in range(steps):
            at = step.to(tl.int64)
            running += tl.load(source + at * in_step, live, 0).to(tl.float64)
            leaving = tl.load(
                source + (at - window) * in_step, live & (step >= window), 0
            )
            running -= leaving.to(tl.float64)
            tl.store(
                target + (at - window + 1) * out_step,
                (running * scale).to(out_ptr.dtype.element_ty),
                live & (step >= window - 1),
            )


# rows per regional program: the running Gram restarts exactly every SEGMENT rows,
# and within a segment wherever a square sum fell below 1 / RESTART of its peak
SEGMENT, RESTART = 32, 64.0


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
        )
    return out


# lanes per window-sum program
WINDOW_LANES = 128


def neighborhood_means(x: torch.Tensor, ky: int, kx: int) -> torch.Tensor:
    """x B,C,H,W (FP16 or FP32, any strides) to its FP32 means over the ky x kx windows
    at every start, B,H-ky+1,W-kx+1,C (channels innermost)."""
    batch, channels, height, width = x.shape
    rows = torch.empty(
        (batch, channels, height - ky + 1, width), dtype=torch.float64, device=x.device
    )
    out = torch.empty(
        (batch, height - ky + 1, width - kx + 1, channels),
        dtype=torch.float32,
        device=x.device,
    )
    with torch.cuda.device(x.device):
        # rows: lanes are the columns, steps the rows
        _window_sums[(triton.cdiv(width, WINDOW_LANES) * batch * channels,)](
            x,
            rows,
            height,
            width,
            channels,
            ky,
            1.0,
            x.stride(0),
            x.stride(1),
            x.stride(3),
            x.stride(2),
            rows.stride(0),
            rows.stride(1),
            rows.stride(3),
            rows.stride(2),
            block=WINDOW_LANES,
        )
        # columns: lanes are the rows of the row sums, steps the columns
        _window_sums[(triton.cdiv(height - ky + 1, WINDOW_LANES) * batch * channels,)](
            rows,
            out,
            width,
            height - ky + 1,
            channels,
            kx,
            1.0 / (ky * kx),
            rows.stride(0),
            rows.stride(1),
            rows.stride(2),
            rows.stride(3),
            out.stride(0),
            out.stride(3),
            out.stride(1),
            out.stride(2),
            block=WINDOW_LANES,
        )
    return out


# positions x channels per routing program
ROUTED_P, ROUTED_C = 64, 32


def routed_sum(
    parts: torch.Tensor, coefficients: torch.Tensor, layout: torch.memory_format
) -> torch.Tensor:
    """parts B,(K+1)*C,H,W (the base, then the experts; FP16 or FP32) and FP32
    coefficients B,K,H,W, each with its H and W flattening to one stride, to
    base + sum_k c_k * expert_k B,C,H,W in the parts' type and the given layout."""
    batch, total, height, width = parts.shape
    experts = coefficients.shape[1]
    channels = total // (experts + 1)
    out = torch.empty(
        (batch, channels, height, width),
        dtype=parts.dtype,
        device=parts.device,
        memory_format=layout,
    )
    grid = (
        triton.cdiv(height * width, ROUTED_P),
        triton.cdiv(channels, ROUTED_C),
        batch,
    )
    with torch.cuda.device(parts.device):
        _routed_sum[grid](
            parts,
            coefficients,
            out,
            height * width,
            channels,
            parts.stride(0),
            parts.stride(1),
            parts.stride(3),
            coefficients.stride(0),
            coefficients.stride(1),
            coefficients.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(3),
            experts=experts,
            block_p=ROUTED_P,
            block_c=ROUTED_C,
            num_warps=4,
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
