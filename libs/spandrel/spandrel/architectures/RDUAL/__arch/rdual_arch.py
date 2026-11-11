"""Recurrent DUAL Light: a carried trunk state fused by one flow-guided stage.

Frames are B,T,3,H,W and the output is B,T,3,sH,sW, with native scales 1/2/4.
Every frame runs the DUAL Light graph once: C128, four heads, group depths
(4,3,3,6) and PixelUnshuffle2 input packing. After the first group it reads
neighboring frames' end-of-trunk features through query-centered temporal
attention (radius 3). A reverse sweep first produces future states, and the
forward sweep reads both neighbors with the same weights.

Guidance is detached, B,T-1,2,2,Htrunk,Wtrunk in target-to-source x/y trunk
pixels. Entry [i,0] maps frame i+1 onto frame i (past donor) and entry [i,1]
maps frame i onto frame i+1 (future donor). Sample LR flow at PU2 feature
centers and divide displacement by two.

The flow estimator is part of the model: a frozen NeuFlow v2 (flow.*, saved
with the checkpoint, never trained, always in eval mode) that guidance()
runs, and that forward(), infer() and backward_clip() use when no guidance is
passed. The flow_loaded buffer records whether it holds real weights;
load_flow_weights() takes an upstream NeuFlow v2 state dict. A guidance
tensor may still be passed instead.

NeuFlow v2 (Zhiyong Zhang, Aniket Gupta, Huaizu Jiang, Hanumant Singh;
https://github.com/neufieldrobotics/NeuFlow_v2) is Apache-2.0 licensed; see
NEUFLOW_LICENSE.txt. Changed from the original: reduced to inference (one
s16 and eight s8 refinement iterations, the 8x upsampling of the last one
only), the single-level correlation lookup calls the native grid sampler
directly, inputs are not divided in place, frames are padded to at least
32 pixels a side (smaller ones gave NaN flow), and the cached grids are
rebuilt whenever the pair batch, frame size or device changes. Parameter
names match the original checkpoint under the prefix flow.

Each donor also carries three alignment cues on the reading frame's trunk
grid, computed once per clip without gradients: photometric error after the
warp, forward-backward flow inconsistency and out-of-frame sampling. A small
convolution turns them into one score bias per head, which joins a learned
source quality of the donor in a reserved lane of every attention head, so
the kernels see an ordinary key. Both bias terms start at zero.

The fusion output starts at zero, so a DUAL Light checkpoint loads with only
temporal.*, flow.* and flow_loaded missing and reproduces DUAL Light on every
frame. Frame validity and shot IDs gate temporal edges.

encode() runs a frame's state-independent part and step_encoded() the rest,
so a clip can be trained frame by frame with one backward per frame, its
frames encoded once and reused. Inside fold_cache() every EDBB kernel is
composed once for the whole clip instead of once per frame, with the same
parameter gradients.

backward_clip() trains one clip with a per-frame loss: its parameter and frame
gradients equal those of forward() followed by one backward, while the clip
runs frame by frame inside fold_cache(). keep trades memory for speed: "none"
recomputes each frame's graph for its backward, "both" ("reverse" for the
reverse sweep only) holds the model's graphs instead. step_functions()
supplies its eager or compiled per-frame callables. For CUDA frames it marks
one CUDA-graph step per call group and frame and keeps everything that
outlives a step in ordinary memory; under CUDA graphs keep="none" is the
supported mode.

infer() runs a long clip without gradients in chunks, carrying the forward
state across them and starting each reverse sweep a look-ahead past the chunk.
prepare_for_export() folds the reparameterized convolutions of an eval copy
for deployment.

Contents:
    Shared helpers (selective checkpointing, initialization, window
    geometry); token normalization, the convolutional FFN and the Fourier
    relative bias; spatial attention (anchored G16 local windows with a dense
    relative bias, context neighborhoods with Flex block masks); cross-scale
    retrieval with its 2x and 4x transfers; regional channel attention and the
    dual-context layer; EDBB reparameterized convolution with its clip fold,
    and the dynamic-expert group convolution; the residual layer and group;
    the optional Triton kernels (direct local attention, temporal attention,
    joint K/V flow warp); the StateFusion temporal stage with its PyTorch
    reference; the frozen NeuFlow v2 flow estimator; RDUAL with its clip
    procedure and the rdual factory.

    Each group follows the slots [local, shifted local, context-small, local,
    shifted local, context-large] and keeps the DUAL Light selection. The
    coarse RGB branch of the retrieval reuses the stem and the first group's
    two local layers. Retrieval follows group two; its 2x transfer always
    feeds the 64-channel reconstruction and its 4x transfer does so at scales
    2 and 4.

Execution:
    Eager execution, evaluation, CPU and export use SDPA throughout.
    prepare_for_compile() selects the training paths: for BF16 or FP16 it
    switches the local layers to the direct kernels when Triton is available,
    and for BF16 on CUDA it folds context images into heads to share masks.
    The default attn_type="auto" resolves to "hybrid" when Flex attention is
    importable, otherwise to "masked"; hybrid then runs the context
    neighborhoods through Flex masks primed for the prepared frame size in
    compiled training outside deterministic mode. A frame size without
    prepared masks runs the masked path; an explicit attn_type="hybrid"
    raises instead, and also when Flex attention is missing. A direct local
    window that cannot run (non-CUDA tensors, partial windows, FP32) falls
    back to SDPA with one warning per layer.

    The default temporal_backend="auto" runs the bundled temporal kernels for
    CUDA features that are not FP64, with FP32 guidance, when Triton and the
    torch.library Triton APIs are available and deterministic algorithms are
    off; otherwise it uses the PyTorch reference. The choice is made on every
    call, in eager and compiled execution alike. "reference" always uses the
    PyTorch path. "triton" requires the kernels on CUDA and raises when
    Triton is missing, deterministic mode is on or a dtype is unsupported;
    CPU tensors always take the reference path. Custom ops are registered
    under rdual_standalone::, so this file loads beside other files that
    bundle similar kernels in one process.

Dependencies:
    PyTorch; optionally Triton with the torch.library Triton APIs and Flex
    attention. No other architecture or local kernel file is required.
"""

# Triton pointer types are inferred; its compile-time arguments use uppercase names.
# ruff: noqa: ANN001, ANN202, N803, N806

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Literal, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.attention import SDPBackend, sdpa_kernel
from spandrel.util import store_hyperparameters
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    create_block_mask = flex_attention = None

_FLEX_ATTENTION_OP = getattr(torch.ops.higher_order, "flex_attention", None)


# Shared initialization, precision and window geometry.


def _selective_checkpoint_policy(
    ctx: Any, op: Any, *args: Any, **kwargs: Any
) -> CheckpointPolicy:
    """Retain attention results while rematerializing the rest of each layer."""
    del ctx, args, kwargs
    saved = (
        torch.ops.aten._scaled_dot_product_flash_attention.default,  # noqa: SLF001
        torch.ops.aten._scaled_dot_product_efficient_attention.default,  # noqa: SLF001
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,  # noqa: SLF001
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,  # noqa: SLF001
    )
    if op in saved or (_FLEX_ATTENTION_OP is not None and op is _FLEX_ATTENTION_OP):
        return CheckpointPolicy.MUST_SAVE
    if (
        _HAS_DIRECT_ATTENTION
        and op is torch.ops.rdual_standalone.dense_rpb_attention.default
    ):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _checkpoint_layer(layer: nn.Module, x: Tensor) -> Tensor:
    """One selective checkpoint boundary, shared by fine and coarse execution."""
    context = nullcontext()
    if not torch.compiler.is_compiling() and torch.is_autocast_enabled(x.device.type):
        # Shared prefix weights may already be in eager AMP's cast cache.
        # Recompute must see the same casts as the recorded forward region.
        context = torch.autocast(
            x.device.type,
            dtype=torch.get_autocast_dtype(x.device.type),
            cache_enabled=False,
        )
    with context:
        return checkpoint(
            layer,
            x,
            use_reentrant=False,
            context_fn=partial(
                create_selective_checkpoint_contexts, _selective_checkpoint_policy
            ),
        )


def _init_weights(module: nn.Module) -> None:
    """Initialize token projections and biases; spatial kernels keep defaults."""
    if isinstance(module, nn.Linear) or (
        isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1)
    ):
        nn.init.trunc_normal_(module.weight, std=0.02)
    if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
        nn.init.zeros_(module.bias)


@dataclass(frozen=True)
class AxisWindows:
    """Gather indices and the unique designated output slot for each pixel."""

    indices: Tensor
    restore: Tensor


def anchored_axis(
    length: int, window: int, offset: int, device: torch.device | str = "cpu"
) -> AxisWindows:
    """Clamp segment windows inward without changing their output ownership.

    Segments start at -offset + k*window. Each real position belongs to its
    segment, but its attention window is anchored inside the image. When an
    axis is shorter than one window, both offsets use one clamped window.
    """
    positions = torch.arange(length, device=device)
    if length <= window:
        return AxisWindows(positions.unsqueeze(0), positions)
    starts = torch.arange(-offset, length, window, device=device)
    starts = starts.clamp(0, length - window)
    owner = (positions + offset) // window
    local = positions - starts.index_select(0, owner)
    indices = starts[:, None] + torch.arange(window, device=device)[None, :]
    return AxisWindows(indices, owner * window + local)


def gather_windows(x: Tensor, rows: AxisWindows, cols: AxisWindows) -> Tensor:
    """BHWC -> (batch*row_windows*column_windows, tokens, channels)."""
    batch, _, _, channels = x.shape
    nr, wh = rows.indices.shape
    nc, ww = cols.indices.shape
    gathered = x.index_select(1, rows.indices.flatten()).index_select(
        2, cols.indices.flatten()
    )
    return (
        gathered.reshape(batch, nr, wh, nc, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch * nr * nc, wh * ww, channels)
    )


def restore_windows(
    windows: Tensor, rows: AxisWindows, cols: AxisWindows, batch: int
) -> Tensor:
    """Gather designated outputs, discarding duplicated overlap queries."""
    nr, wh = rows.indices.shape
    nc, ww = cols.indices.shape
    channels = windows.shape[-1]
    expanded = (
        windows.reshape(batch, nr, nc, wh, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, nr * wh, nc * ww, channels)
    )
    return expanded.index_select(1, rows.restore).index_select(2, cols.restore)


def relative_position_index(
    height: int, width: int, nominal_window: int, device: torch.device
) -> Tensor:
    """Swin query-minus-key row-major table index, including rectangular crops."""
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    y, x = yy.flatten(), xx.flatten()
    stride = 2 * nominal_window - 1
    return (y[:, None] - y[None, :] + nominal_window - 1) * stride + (
        x[:, None] - x[None, :] + nominal_window - 1
    )


def calculation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use FP32 for low-precision stability, preserve FP64 reference tests."""
    return torch.float64 if dtype == torch.float64 else torch.float32


# Token-wise normalization, feed-forward processing and positional bias.


class TokenNorm(nn.Module):
    """Channel-wise LayerNorm at each pixel, with FP32 low-precision statistics."""

    def __init__(self, channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        precision = calculation_dtype(x.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = F.layer_norm(
                x.permute(0, 2, 3, 1).to(precision),
                self.norm.normalized_shape,
                self.norm.weight.to(precision),
                self.norm.bias.to(precision),
                self.norm.eps,
            )
        return result.permute(0, 3, 1, 2).to(x.dtype)


class ConvFFN(nn.Module):
    """Expansion-two FFN with GELU, a depthwise residual and a 1x1 readout."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = 2 * channels
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.project = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.gelu(self.expand(x))
        return self.project(hidden + F.gelu(self.depthwise(hidden)))


class RelativeFourierBias(nn.Module):
    """Factor a stationary signed relative bias without constructing N*N.

    Dot(q_factor(i), k_factor(j)) = sum(a*cos(w*(i-j))+b*sin(w*(i-j))).
    Frequency and amplitude shapes are independent of the attention window.
    """

    rank = 16

    def __init__(self, heads: int) -> None:
        super().__init__()
        count = self.rank // 2
        magnitudes = torch.logspace(
            math.log10(2 * math.pi / 128), math.log10(math.pi / 2), count
        )
        angles = torch.arange(count) * (math.pi * (3 - math.sqrt(5)))
        self.frequency = nn.Parameter(
            magnitudes[:, None] * torch.stack((angles.cos(), angles.sin()), dim=-1)
        )
        self.amplitude_cos = nn.Parameter(torch.full((heads, count), 1 / count))
        self.amplitude_sin = nn.Parameter(torch.zeros(heads, count))

    def factors(self, coordinates: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        precision = calculation_dtype(dtype)
        with torch.autocast(device_type=self.frequency.device.type, enabled=False):
            phase = coordinates.to(precision) @ self.frequency.to(precision).T
            cosine, sine = phase.cos()[None], phase.sin()[None]
            a = self.amplitude_cos.to(precision)[:, None]
            b = self.amplitude_sin.to(precision)[:, None]
            query = torch.cat((a * cosine + b * sine, a * sine - b * cosine), dim=-1)
            key = torch.cat((cosine, sine), dim=-1).expand(query.shape[0], -1, -1)
        return query.to(dtype), key.to(dtype)

    def forward(
        self, height: int, width: int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=self.frequency.device),
            torch.arange(width, device=self.frequency.device),
            indexing="ij",
        )
        return self.factors(torch.stack((yy, xx), dim=-1).reshape(-1, 2), dtype)


# Spatial attention: anchored windows and sliding context neighborhoods.


def _na_coordinates(height: int, width: int, device: torch.device) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


@lru_cache(maxsize=16)
def _inference_na_coordinates(height: int, width: int, device: torch.device) -> Tensor:
    """Reuse small, read-only integer grids, never weights or attention masks.

    Normal tensors remain safe if a later caller needs autograd. Training and
    compiled/export paths bypass this bounded eager-inference cache.
    """
    with torch.inference_mode(False):
        return _na_coordinates(height, width, device)


def _na_block_mask(
    height: int, width: int, window: int, first: int, device: torch.device
) -> Any:
    """Build one immutable four-tile Flex mask outside the compiled graph.

    Rows use clamped query coordinates and inward-shifted donor origins, exactly
    like _na_chunk. Image batches share this geometry through the head axis.
    The callback captures integer geometry only, never learned Fourier factors.
    """
    kh, kw = min(height, window), min(width, window)
    uh, uw = min(height, kh + 15), min(width, kw + 15)
    tiles = [(y, x) for y in range(0, height, 16) for x in range(0, width, 16)]
    current = tiles[first : first + 4]
    with torch.inference_mode(False), torch.no_grad():
        origins = torch.tensor(current, device=device, dtype=torch.int64)

        def mask_mod(b, h, q, k):
            oy, ox = origins[b, 0], origins[b, 1]
            qy = (oy + q // 16).clamp(max=height - 1)
            qx = (ox + q % 16).clamp(max=width - 1)
            ky = (oy - kh // 2).clamp(0, height - uh) + k // uw
            kx = (ox - kw // 2).clamp(0, width - uw) + k % uw
            sy = (qy - kh // 2).clamp(0, height - kh)
            sx = (qx - kw // 2).clamp(0, width - kw)
            return (
                (q < 256)
                & (k < uh * uw)
                & (ky >= sy)
                & (ky < sy + kh)
                & (kx >= sx)
                & (kx < sx + kw)
            )

        return create_block_mask(
            mask_mod,
            len(current),
            None,
            256,
            uh * uw,
            device=device,
            BLOCK_SIZE=128,
        )


class _SpatialAttention(nn.Module):
    """Spatial attention on normalized BCHW features, gated and projected.

    Local layers attend in anchored windows, shifted by half a window when
    shifted, with a dense relative position bias. Context layers carry the
    Fourier relative bias instead; once the image exceeds their window they
    gather one in-bounds neighborhood per query, shifted inward at image
    borders. Their queries run in 16x16 tiles, four tiles per chunk, and a
    per-query mask enforces the neighborhood although a chunk shares donors.
    Normalization and the residual addition belong to the enclosing layer.
    """

    def __init__(
        self, channels: int, heads: int, window: int, *, shifted: bool, context: bool
    ) -> None:
        super().__init__()
        self.channels, self.heads, self.window = channels, heads, window
        self.offset = window // 2 if shifted else 0
        self.context = context
        self.qkv = nn.Linear(channels, 3 * channels)
        self.relative_bias: nn.Parameter | None = None
        self.factored_bias: RelativeFourierBias | None = None
        if not context:
            self.relative_bias = nn.Parameter(torch.empty((2 * window - 1) ** 2, heads))
            nn.init.trunc_normal_(self.relative_bias, std=0.02)
        self.gate_depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.gate_pointwise = nn.Conv2d(channels, channels, 1)
        self.project = nn.Linear(channels, channels)
        if context:
            self.factored_bias = RelativeFourierBias(heads)
        # Execution state, set by prepare_for_compile() and prepare_for_export().
        self.g_attention_impl: Literal["sdpa", "direct"] = "sdpa"
        self.na_training_fast = False
        self.attn_type: Literal["masked", "hybrid"] = "masked"
        self.require_hybrid = False
        self.flex_masks: dict[tuple[int, int, int], Any] = {}
        self.direct_fallback_reason: str | None = None

    def _apply(self, fn: Any, recurse: bool = True) -> _SpatialAttention:
        # BlockMasks are runtime geometry, not buffers or checkpoint contents.
        # Device/dtype moves require fresh preparation on the destination.
        self.flex_masks.clear()
        return super()._apply(fn, recurse=recurse)

    def _record_direct_fallback(self, reason: str) -> None:
        if self.direct_fallback_reason is None:
            self.direct_fallback_reason = reason
            warnings.warn(
                f"RDUAL G{self.window} direct attention falls back to SDPA: {reason}",
                stacklevel=3,
            )

    def _use_fast_na(self, query: Tensor) -> bool:
        """Use the shared-mask layout only on its supported training backend."""
        return (
            self.na_training_fast
            and self.training
            and query.device.type == "cuda"
            and query.dtype == torch.bfloat16
            and not torch.are_deterministic_algorithms_enabled()
        )

    def position_bias(
        self, height: int, width: int, dtype: torch.dtype
    ) -> Tensor | None:
        """The dense relative position bias of a local window, None in context."""
        if self.relative_bias is None:
            return None
        index = relative_position_index(
            height, width, self.window, self.relative_bias.device
        )
        precision = calculation_dtype(dtype)
        values = self.relative_bias.to(precision)[index.flatten()]
        return (
            values.reshape(height * width, height * width, self.heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype)
        )

    def attend_windows(self, tokens: Tensor, window_shape: tuple[int, int]) -> Tensor:
        """Attention within windows of window_shape, before the gate."""
        batch_windows, count, channels = tokens.shape
        head_dim = channels // self.heads
        qkv = self.qkv(tokens).reshape(batch_windows, count, 3, self.heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = query * (head_dim**-0.5)
        if self.g_attention_impl == "direct":
            if tokens.device.type != "cuda" or window_shape != (
                self.window,
                self.window,
            ):
                self._record_direct_fallback("requires CUDA and full windows")
            elif query.dtype not in (torch.bfloat16, torch.float16):
                self._record_direct_fallback("requires BF16 or FP16 attention inputs")
            else:
                assert self.relative_bias is not None
                output = direct_attention(
                    query.contiguous(),
                    key.contiguous(),
                    value.contiguous(),
                    self.relative_bias,
                    self.window,
                )
                return output.transpose(1, 2).reshape(batch_windows, count, channels)
        dtype = query.dtype
        if self.factored_bias is not None:
            bq, bk = self.factored_bias(*window_shape, dtype)
            query = torch.cat(
                (query, bq.unsqueeze(0).expand(batch_windows, -1, -1, -1)), dim=-1
            )
            key = torch.cat(
                (key, bk.unsqueeze(0).expand(batch_windows, -1, -1, -1)), dim=-1
            )
            value = F.pad(value, (0, self.factored_bias.rank))
        bias = self.position_bias(*window_shape, dtype)
        output = F.scaled_dot_product_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            attn_mask=None if bias is None else bias.contiguous(),
            dropout_p=0.0,
            scale=1.0,
        )[..., :head_dim]
        return output.transpose(1, 2).reshape(batch_windows, count, channels)

    def _na_chunk(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        first: int,
        height: int,
        width: int,
        key_bias: Tensor | None = None,
    ) -> Tensor:
        """Gather, mask and attend a query chunk; return unassembled tile outputs."""
        batch, heads, _, dim = query.shape
        channels = heads * dim
        tile, chunk_size = 16, 4
        kh, kw = min(height, self.window), min(width, self.window)
        uh, uw = min(height, kh + tile - 1), min(width, kw + tile - 1)
        eager_inference = (
            not self.training
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and not torch.onnx.is_in_onnx_export()
        )
        coordinates = _inference_na_coordinates if eager_inference else _na_coordinates
        qlocal = coordinates(tile, tile, query.device)
        klocal = coordinates(uh, uw, query.device)
        tiles = [(y, z) for y in range(0, height, tile) for z in range(0, width, tile)]
        current = tiles[first : first + chunk_size]
        count = len(current)
        dtype = query.dtype
        # Eligible BF16 training folds images into heads to share masks.
        # Evaluation, deterministic mode and other dtypes use the standard layout;
        # both paths retain the same neighborhoods and positional coordinates.
        fast = self._use_fast_na(query)

        def gather(t: Tensor, index: Tensor) -> Tensor:
            gathered = t.index_select(2, index.reshape(-1)).reshape(
                batch, heads, count, index.shape[1], dim
            )
            if fast:
                return gathered.permute(2, 0, 1, 3, 4).reshape(
                    count, batch * heads, index.shape[1], dim
                )
            return gathered.permute(0, 2, 1, 3, 4).reshape(
                batch * count, heads, index.shape[1], dim
            )

        if eager_inference:
            # Avoid two tiny host-to-device copies and their synchronizations
            # per chunk. Tile ownership and integer coordinates are unchanged.
            origins = (
                coordinates(
                    (height + tile - 1) // tile,
                    (width + tile - 1) // tile,
                    query.device,
                )[first : first + count]
                * tile
            )
        else:
            origins = torch.tensor(current, device=query.device)
        qp = origins[:, None, :] + qlocal[None]
        qp = torch.stack(
            (qp[..., 0].clamp(max=height - 1), qp[..., 1].clamp(max=width - 1)), -1
        )
        if eager_inference:
            ko = torch.stack(
                (
                    (origins[:, 0] - kh // 2).clamp(0, height - uh),
                    (origins[:, 1] - kw // 2).clamp(0, width - uw),
                ),
                -1,
            )
        else:
            ko = origins - torch.tensor((kh // 2, kw // 2), device=query.device)
            ko = torch.stack(
                (ko[:, 0].clamp(0, height - uh), ko[:, 1].clamp(0, width - uw)), -1
            )
        kp = ko[:, None, :] + klocal[None]
        qi, ki = qp[..., 0] * width + qp[..., 1], kp[..., 0] * width + kp[..., 1]
        qt, kt, vt = gather(query, qi), gather(key, ki), gather(value, ki)
        bq, _ = self.factored_bias.factors((qp - ko[:, None]).reshape(-1, 2), dtype)
        if key_bias is None:
            _, bk = self.factored_bias.factors(klocal, dtype)
        else:
            bk = key_bias
        rank = self.factored_bias.rank
        bq = bq.reshape(heads, count, tile * tile, rank).permute(1, 0, 2, 3)
        if fast:
            bq = (
                bq[:, None]
                .expand(count, batch, heads, tile * tile, rank)
                .reshape(count, batch * heads, tile * tile, rank)
            )
            bk = (
                bk[None, None]
                .expand(count, batch, heads, uh * uw, rank)
                .reshape(count, batch * heads, uh * uw, rank)
            )
        else:
            bq = (
                bq[None]
                .expand(batch, -1, -1, -1, -1)
                .reshape(batch * count, heads, tile * tile, rank)
            )
            bk = bk[None].expand(batch * count, -1, -1, -1)
        qt, kt = (
            torch.cat((qt, bq), -1).contiguous(),
            torch.cat((kt, bk), -1).contiguous(),
        )
        # Positional factors extend Q/K only. Zero V lanes never contribute to
        # the returned channels; the supported training backend accepts Dv!=Dq.
        vt = vt.contiguous() if fast else F.pad(vt, (0, rank)).contiguous()
        block = None
        if (
            fast
            and self.attn_type == "hybrid"
            and torch.compiler.is_compiling()
            and not torch.onnx.is_in_onnx_export()
        ):
            block = self.flex_masks.get((height, width, first))
            if block is None and self.require_hybrid:
                raise RuntimeError(
                    "RDUAL hybrid mask missing: call prepare_for_compile with the "
                    "actual input shape and dtype after moving the model to CUDA"
                )
        if block is not None:
            # Content32 + RFB16 becomes64 by appending exact zeros. V stays32;
            # learned RFB factors remain ordinary differentiable Q/K operands.
            padded = 1 << (qt.shape[-1] - 1).bit_length()
            attended = flex_attention(
                F.pad(qt, (0, padded - qt.shape[-1])),
                F.pad(kt, (0, padded - kt.shape[-1])),
                vt,
                block_mask=block,
                scale=1.0,
            )
            return (
                attended.reshape(count, batch, heads, tile, tile, dim)
                .permute(1, 0, 2, 5, 3, 4)
                .reshape(batch, count, channels, tile, tile)
            )
        sy = (qp[..., 0] - kh // 2).clamp(0, height - kh)
        sx = (qp[..., 1] - kw // 2).clamp(0, width - kw)
        ky = ko[:, 0, None] + torch.arange(uh, device=query.device)[None]
        kx = ko[:, 1, None] + torch.arange(uw, device=query.device)[None]
        my = (ky[:, None, :] >= sy[..., None]) & (ky[:, None, :] < sy[..., None] + kh)
        mx = (kx[:, None, :] >= sx[..., None]) & (kx[:, None, :] < sx[..., None] + kw)
        mask = (my[..., :, None] & mx[..., None, :]).reshape(
            count, tile * tile, uh * uw
        )
        if fast:
            # Fold independent images into heads to share each chunk's mask.
            # Prefer cuDNN, with ordinary SDPA capability dispatch on devices
            # where it is unsupported; do not swallow kernel/runtime errors.
            with sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ],
                set_priority=True,
            ):
                attended = F.scaled_dot_product_attention(
                    qt,
                    kt,
                    vt,
                    attn_mask=mask[:, None].contiguous(),
                    dropout_p=0.0,
                    scale=1.0,
                )
            return (
                attended.reshape(count, batch, heads, tile, tile, dim)
                .permute(1, 0, 2, 5, 3, 4)
                .reshape(batch, count, channels, tile, tile)
            )
        mask = (
            mask[None]
            .expand(batch, -1, -1, -1)
            .reshape(batch * count, 1, tile * tile, uh * uw)
            .contiguous()
        )
        attended = F.scaled_dot_product_attention(
            qt, kt, vt, attn_mask=mask, dropout_p=0.0, scale=1.0
        )[..., :dim]
        return (
            attended.reshape(batch, count, heads, tile, tile, dim)
            .permute(0, 1, 2, 5, 3, 4)
            .reshape(batch, count, channels, tile, tile)
        )

    def _na_core(self, x: Tensor) -> Tensor:
        """Evaluate exact per-query neighborhoods in bounded spatial chunks."""
        batch, channels, height, width = x.shape
        heads, dim = self.heads, channels // self.heads
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        qkv = self.qkv(tokens).reshape(batch, height * width, 3, heads, dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = query * dim**-0.5
        output = x.new_empty((batch, channels, height, width), dtype=query.dtype)
        tile, chunk_size = 16, 4
        tiles = [(y, z) for y in range(0, height, tile) for z in range(0, width, tile)]
        key_bias = None
        if (
            not self.training
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and not torch.onnx.is_in_onnx_export()
        ):
            # Every chunk in this layer uses the same local key coordinates.
            # Recompute once PER FORWARD so EMA/load updates cannot leave stale
            # positional factors. Arithmetic, dtype and SDPA backend stay intact.
            kh, kw = min(height, self.window), min(width, self.window)
            uh, uw = min(height, kh + tile - 1), min(width, kw + tile - 1)
            _, key_bias = self.factored_bias.factors(
                _inference_na_coordinates(uh, uw, query.device), query.dtype
            )
        for first in range(0, len(tiles), chunk_size):
            attended = self._na_chunk(query, key, value, first, height, width, key_bias)
            for j, (y, z) in enumerate(tiles[first : first + chunk_size]):
                th, tw = min(tile, height - y), min(tile, width - z)
                output[:, :, y : y + th, z : z + tw] = attended[:, j, :, :th, :tw]
        return output

    def pre_gate_core(self, x: Tensor) -> Tensor:
        """BCHW attention output, excluding the gate and projection."""
        batch, channels, height, width = x.shape
        if self.context and (height > self.window or width > self.window):
            return self._na_core(x)
        wh, ww = min(height, self.window), min(width, self.window)
        if self.offset == 0 and height % wh == 0 and width % ww == 0:
            nr, nc = height // wh, width // ww
            tokens = (
                x.permute(0, 2, 3, 1)
                .reshape(batch, nr, wh, nc, ww, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch * nr * nc, wh * ww, channels)
            )
            output = self.attend_windows(tokens, (wh, ww))
            return (
                output.reshape(batch, nr, nc, wh, ww, channels)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(batch, height, width, channels)
                .permute(0, 3, 1, 2)
            )
        rows = anchored_axis(height, self.window, self.offset, x.device)
        cols = anchored_axis(width, self.window, self.offset, x.device)
        tokens = gather_windows(x.permute(0, 2, 3, 1), rows, cols)
        output = self.attend_windows(
            tokens, (rows.indices.shape[1], cols.indices.shape[1])
        )
        return restore_windows(output, rows, cols, batch).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        attended = self.pre_gate_core(x)
        gate = self.gate_pointwise(self.gate_depthwise(x)).sigmoid()
        return self.project((attended * gate).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# Cross-scale retrieval: block packing, attention and reconstruction transfers.


def axis_spans(length: int, window: int) -> tuple[tuple[int, int, int], ...]:
    """(start, stop, window extent); each input position is owned exactly once."""
    if length <= window:
        return ((0, length, length),)
    end = (length // window) * window
    spans = [(0, end, window)]
    if end < length:
        spans.append((end, length, length - end))
    return tuple(spans)


def block_layout(height: int, width: int, window: int) -> tuple:
    """Row-major rectangular blocks with equal-sized windows in each block."""
    return tuple(
        (y0, y1, x0, x1, wh, ww)
        for y0, y1, wh in axis_spans(height, window)
        for x0, x1, ww in axis_spans(width, window)
    )


def pack_block(x: Tensor, block: tuple[int, ...]) -> Tensor:
    y0, y1, x0, x1, wh, ww = block
    batch, channels = x.shape[:2]
    return (
        x[:, :, y0:y1, x0:x1]
        .reshape(batch, channels, (y1 - y0) // wh, wh, (x1 - x0) // ww, ww)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(-1, wh * ww, channels)
    )


def unpack_block(tokens: Tensor, block: tuple[int, ...], batch: int) -> Tensor:
    y0, y1, x0, x1, wh, ww = block
    channels = tokens.shape[-1]
    return (
        tokens.reshape(batch, (y1 - y0) // wh, (x1 - x0) // ww, wh, ww, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, y1 - y0, x1 - x0)
    )


def join_blocks(outputs: list[Tensor], layout: tuple) -> Tensor:
    rows, current = [], []
    previous = layout[0][0]
    for result, block in zip(outputs, layout, strict=True):
        if block[0] != previous:
            rows.append(torch.cat(current, dim=3))
            current = []
            previous = block[0]
        current.append(result)
    rows.append(torch.cat(current, dim=3))
    return torch.cat(rows, dim=2)


def grouped_layout(layout: tuple) -> tuple:
    """Batch equal window shapes; never cache tensor outputs across forwards."""
    groups = {}
    for index, block in enumerate(layout):
        groups.setdefault(block[-2:], []).append((index, block))
    return tuple(groups.items())


def pad_even(x: Tensor) -> Tensor:
    """Replicate the last row/column when needed for 2x pixel packing."""
    return F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")


def pack_children(x: Tensor) -> Tensor:
    """B,8*heads,H,W -> B,32*heads,H/2,W/2; head/slot/channel order."""
    b, c, h, w = x.shape
    return (
        F.pixel_unshuffle(x, 2)
        .reshape(b, c // 8, 8, 4, h // 2, w // 2)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, 4 * c, h // 2, w // 2)
    )


def unpack_children(x: Tensor) -> Tensor:
    """Canonical head/slot/c retrieval -> B,8*heads,2H,2W."""
    b, c, h, w = x.shape
    slots = (
        x.reshape(b, c // 32, 4, 8, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, c, h, w)
    )
    return F.pixel_shuffle(slots, 2)


def pack_grandchildren(x: Tensor) -> Tensor:
    """B,2*heads,H,W -> B,32*heads,H/4,W/4; head/slot/channel order."""
    b, c, h, w = x.shape
    return (
        F.pixel_unshuffle(x, 4)
        .reshape(b, c // 2, 2, 16, h // 4, w // 4)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, 16 * c, h // 4, w // 4)
    )


def unpack_grandchildren(x: Tensor) -> Tensor:
    """Restore the sixteen packed spatial slots at four times the resolution."""
    b, c, h, w = x.shape
    raw = x.reshape(b, c // 32, 16, 2, h, w).permute(0, 1, 3, 2, 4, 5)
    return F.pixel_shuffle(raw.reshape(b, c, h, w), 4)


def pack_qk(q: Tensor, k: Tensor) -> tuple:
    """Pair 96px query blocks with half-resolution keys, batching equal shapes."""
    layout = block_layout(*q.shape[-2:], 96)
    classes = []
    for _, items in grouped_layout(layout):
        qs = [pack_block(q, block) for _, block in items]
        ks = [pack_block(k, tuple(t // 2 for t in block)) for _, block in items]
        qq, kk = (
            torch.cat(z, 0)
            .reshape(-1, z[0].shape[1], q.shape[1] // 32, 32)
            .transpose(1, 2)
            for z in (qs, ks)
        )
        classes.append((items, qq, kk.contiguous()))
    return layout, classes, q.shape[0]


def attend_packed(packed: tuple, v: Tensor, log_tau: Tensor) -> Tensor:
    """Reuse packed query/key blocks for a value stream, then restore BCHW."""
    layout, classes, batch = packed
    outputs = [None] * len(layout)
    for items, qq, kk in classes:
        vs = [pack_block(v, tuple(t // 2 for t in block)) for _, block in items]
        vv = (
            torch.cat(vs, 0)
            .reshape(-1, vs[0].shape[1], qq.shape[1], 32)
            .transpose(1, 2)
        )
        tau = log_tau.exp().clamp(0.1, 10).to(qq.dtype)
        out = (
            F.scaled_dot_product_attention(
                (qq * tau[None, :, None, None]).contiguous(),
                kk,
                vv.contiguous(),
                scale=math.log(kk.shape[2]) / (math.log(1024) * math.sqrt(32)),
            )
            .transpose(1, 2)
            .reshape(-1, qq.shape[2], 32 * qq.shape[1])
        )
        start = 0
        for (index, block), chunk in zip(items, vs, strict=True):
            count = chunk.shape[0]
            outputs[index] = unpack_block(out[start : start + count], block, batch)
            start += count
    return join_blocks(outputs, layout)


class CrossScaleRetrieval(nn.Module):
    """Match fine features to a coarse RGB branch and retrieve packed values.

    The coarse branch uses a fixed separable downsampler and shares the stem
    and first two local layers with the fine branch. Queries combine fine-prefix
    and group-two features; keys use the coarse branch, while values carry
    group-two features packed into four spatial slots. Separate zero-initialized
    readouts inject the result into the trunk and the 2x reconstruction stream.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels, self.heads = channels, channels // 32
        self.norm_f = TokenNorm(channels, 1e-5)
        self.norm_u = TokenNorm(channels, 1e-5)
        self.query = nn.Conv2d(2 * channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels // 4, 1)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.project = nn.Conv2d(channels, channels, 1)
        self.hr = nn.Conv2d(channels // 4, 64, 1)
        self.log_tau = nn.Parameter(torch.zeros(self.heads))
        self.register_buffer(
            "down_taps",
            torch.tensor(
                [
                    -0.01171875,
                    -0.03515625,
                    0.11328125,
                    0.43359375,
                    0.43359375,
                    0.11328125,
                    -0.03515625,
                    -0.01171875,
                ]
            ),
            persistent=False,
        )
        self.apply(_init_weights)
        for layer in (self.project, self.hr):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def downsample(self, shifted: Tensor) -> Tensor:
        x = pad_even(shifted)
        taps = self.down_taps.to(x.dtype)
        h = taps.view(1, 1, 1, 8).expand(3, -1, -1, -1)
        v = taps.view(1, 1, 8, 1).expand(3, -1, -1, -1)
        x = F.conv2d(
            F.pad(x, (3, 3, 0, 0), mode="replicate"), h, stride=(1, 2), groups=3
        )
        return F.conv2d(
            F.pad(x, (0, 0, 3, 3), mode="replicate"), v, stride=(2, 1), groups=3
        )


class SecondOctaveTransfer(nn.Module):
    """Reuse cross-scale matching to add a zero-initialized 4x RGB residual.

    Sixteen spatial slots are packed per key. At scale 4 the model emits
    twelve channels here and pixel-shuffles them once more for its internal 8x
    output.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.value = nn.Conv2d(64, channels // 16, 1)
        self.out = nn.Conv2d(channels // 16, 3, 3, padding=1, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(channels // 32))
        self.apply(_init_weights)
        nn.init.zeros_(self.out.weight)

    def forward(self, t2: Tensor, packed: tuple, shape: tuple[int, int]) -> Tensor:
        h, w = shape
        g = F.pad(self.value(t2), (0, 2 * (w % 2), 0, 2 * (h % 2)), mode="replicate")
        o2 = attend_packed(packed, pack_grandchildren(g), self.log_tau)
        r = unpack_grandchildren(o2)[:, :, : 4 * h, : 4 * w]
        return self.out(r)


# Regional channel mixing and the dual-context residual layer.


def hat_weights(
    length: int,
    shift: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Nominal-window hats on real pixels, including clipped border windows."""
    t = (torch.arange(length, device=device) + shift) % 32
    return (1 - ((t.to(dtype) - 15.5).abs() / 16)).to(dtype)


class RegionalChannelAttention(nn.Module):
    """Channel attention over four offset grids of 32x32 regions.

    Each region forms a per-head channel-mixing matrix from normalized feature
    correlations. Separable hat weights blend the four grid outputs. Border
    normalization counts real pixels, not padding; the caller adds the residual.
    """

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm = TokenNorm(channels, eps=1e-5)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.depthwise = nn.Conv2d(
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.theta = nn.Parameter(torch.zeros(heads))
        self.apply(_init_weights)

    def channel_attention(self, q: Tensor, k: Tensor, count: Tensor) -> Tensor:
        """(..., heads, d, tokens) -> (..., heads, d, d), FP32 or FP64.

        `count` is the number of REAL tokens, never the padded window area.
        Normalize after the raw Gram to avoid low-precision normalized operands.
        """
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type, enabled=False):
            qf, kf = q.to(dtype), k.to(dtype)
            count = count.to(dtype)
            qnorm = (qf.square().sum(-1) + count * 1e-6).sqrt()
            knorm = (kf.square().sum(-1) + count * 1e-6).sqrt()
            gram = qf @ kf.transpose(-2, -1)
            cosine = gram / (qnorm[..., :, None] * knorm[..., None, :])
            tau = self.theta.to(dtype).clamp(max=math.log(100)).exp()
            return (cosine * tau[:, None, None]).softmax(-1)

    def _class_core(self, q: Tensor, k: Tensor, v: Tensor, sy: int, sx: int) -> Tensor:
        b, c, h, w = q.shape
        ph, pw = h + sy + (-(h + sy) % 32), w + sx + (-(w + sx) % 32)
        ny, nx, d = ph // 32, pw // 32, c // self.heads

        def windows(t: Tensor) -> Tensor:
            t = F.pad(t, (sx, pw - w - sx, sy, ph - h - sy))
            return (
                t.reshape(b, self.heads, d, ny, 32, nx, 32)
                .permute(0, 3, 5, 1, 2, 4, 6)
                .reshape(b, ny * nx, self.heads, d, 1024)
            )

        # Real clipped lengths rather than a stored pixel-validity tensor.
        y0 = torch.arange(ny, device=q.device) * 32 - sy
        x0 = torch.arange(nx, device=q.device) * 32 - sx
        yc = (y0 + 32).clamp(max=h) - y0.clamp(min=0)
        xc = (x0 + 32).clamp(max=w) - x0.clamp(min=0)
        count = (yc[:, None] * xc[None, :]).reshape(1, ny * nx, 1, 1)
        a = self.channel_attention(windows(q), windows(k), count)
        # Preserve the probability cast before the low-precision value product.
        # Keep matmul dtype explicit even under a surrounding autocast context.
        with torch.autocast(device_type=v.device.type, enabled=False):
            o = a.to(v.dtype) @ windows(v)
            o = (
                o.reshape(b, ny, nx, self.heads, d, 32, 32)
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(b, c, ph, pw)[:, :, sy : sy + h, sx : sx + w]
                .to(a.dtype)
            )
            wy = hat_weights(h, sy, device=q.device, dtype=a.dtype)
            wx = hat_weights(w, sx, device=q.device, dtype=a.dtype)
            return o * wy[None, None, :, None] * wx[None, None, None, :]

    def mix(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Blend four class outputs without an H*W*d*d matrix field."""
        result = None
        for sy, sx in ((0, 0), (0, 16), (16, 0), (16, 16)):
            out = self._class_core(q, k, v, sy, sx)
            result = out if result is None else result + out
        return result

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.depthwise(self.qkv(self.norm(x))).chunk(3, dim=1)
        return self.project(self.mix(q, k, v).to(x.dtype))


class DualContextLayer(nn.Module):
    """Spatial attention -> regional channel mixing -> convolutional FFN."""

    def __init__(self, source: _Layer) -> None:
        super().__init__()
        self.norm1, self.attention = source.norm1, source.attention
        self.norm2, self.ffn = source.norm2, source.ffn
        self.dual = RegionalChannelAttention(
            self.attention.channels,
            self.attention.heads,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.dual(x)
        return x + self.ffn(self.norm2(x))


# Reparameterized convolutions and per-image expert routing.


class _EDBBDenseBranch(nn.Module):
    """Independent gain-2 1x1 -> 3x3 -> 1x1 factors, never applied to images."""

    def __init__(self, ci: int, co: int, *, bias: bool, factory: dict) -> None:
        super().__init__()
        self.expand = nn.Conv2d(ci, 2 * ci, 1, bias=bias, **factory)
        self.spatial = nn.Conv2d(2 * ci, 2 * co, 3, bias=bias, **factory)
        self.reduce = nn.Conv2d(2 * co, co, 1, bias=bias, **factory)

    def equivalent_kernel(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        expand = self.expand.weight[:, :, 0, 0].to(dtype)
        spatial = self.spatial.weight.to(dtype)
        reduce = self.reduce.weight[:, :, 0, 0].to(dtype)
        projected_spatial = torch.einsum("om,mnxy->onxy", reduce, spatial)
        kernel = torch.einsum("omxy,mi->oixy", projected_spatial, expand)
        bias = None
        if self.expand.bias is not None:
            bias = reduce @ (
                spatial.sum((-2, -1)) @ self.expand.bias.to(dtype)
                + self.spatial.bias.to(dtype)
            ) + self.reduce.bias.to(dtype)
        return kernel, bias


class _EDBBPointwiseBranch(nn.Module):
    """Gain-2 pointwise factors; their effective kernel remains 1x1."""

    def __init__(self, ci: int, co: int, *, bias: bool, factory: dict) -> None:
        super().__init__()
        self.expand = nn.Conv2d(ci, 2 * ci, 1, bias=bias, **factory)
        self.reduce = nn.Conv2d(2 * ci, co, 1, bias=bias, **factory)

    def equivalent_kernel(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        reduce = self.reduce.weight[:, :, 0, 0].to(dtype)
        kernel = (reduce @ self.expand.weight[:, :, 0, 0].to(dtype))[:, :, None, None]
        bias = None
        if self.expand.bias is not None:
            bias = reduce @ self.expand.bias.to(dtype) + self.reduce.bias.to(dtype)
        return kernel, bias


class _EDBBEdgeBranch(nn.Module):
    """Expanded pointwise -> scaled depthwise edge filter -> output projection."""

    def __init__(
        self, ci: int, co: int, mask: Tensor, *, bias: bool, factory: dict
    ) -> None:
        super().__init__()
        # A projection bias cancels against the zero-sum bias-padded filter.
        self.project = nn.Conv2d(ci, 2 * co, 1, bias=False, **factory)
        self.scale = nn.Parameter(torch.zeros(2 * co, 1, 1, 1, **factory))
        self.bias = nn.Parameter(torch.zeros(2 * co, **factory)) if bias else None
        self.reduce = nn.Conv2d(2 * co, co, 1, bias=bias, **factory)
        self.register_buffer("mask", mask.reshape(1, 1, 3, 3), persistent=False)

    def equivalent_kernel(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        reduce = self.reduce.weight[:, :, 0, 0].to(dtype)
        expanded = (
            self.project.weight.to(dtype) * self.scale.to(dtype) * self.mask.to(dtype)
        )
        kernel = torch.einsum("om,mixy->oixy", reduce, expanded)
        bias = None
        if self.bias is not None:
            bias = reduce @ self.bias.to(dtype) + self.reduce.bias.to(dtype)
        return kernel, bias


class EDBBReparamConv3x3(nn.Module):
    """Compose dense, pointwise, edge and identity branches into one 3x3 kernel.

    Both training and evaluation compose parameters before convolution; expanded
    intermediate image features are not materialized. Composition uses FP32
    (FP64 for double parameters), then casts back before convolution. Branch
    initialization exactly embeds the supplied convolution. folded() produces
    a static deployment convolution without changing the training instance.

    While fold_active, FP32 composition reads a clip-level kernel and bias
    instead (RDUAL.fold_cache()). They are non-persistent buffers refreshed in
    place, so a compiled frame keeps seeing the same tensors on every clip, and
    require gradients only while a tracking block is open.
    """

    fold_active: bool
    fold_kernel: Tensor | None
    fold_bias: Tensor | None

    def __init__(self, original: nn.Conv2d) -> None:
        super().__init__()
        ci, co = original.in_channels, original.out_channels
        self.in_channels, self.out_channels = ci, co
        bias = original.bias is not None
        factory = {"device": original.weight.device, "dtype": original.weight.dtype}
        # The two dense paths have identical topology but independent factors.
        self.direct = _EDBBDenseBranch(ci, co, bias=bias, factory=factory)
        self.sequential = _EDBBDenseBranch(ci, co, bias=bias, factory=factory)
        self.pointwise = _EDBBPointwiseBranch(ci, co, bias=bias, factory=factory)
        masks = (
            ((1, 0, -1), (2, 0, -2), (1, 0, -1)),
            ((1, 2, 1), (0, 0, 0), (-1, -2, -1)),
            ((0, 1, 0), (1, -4, 1), (0, 1, 0)),
        )
        self.edges = nn.ModuleList(
            _EDBBEdgeBranch(
                ci, co, torch.tensor(mask, **factory), bias=bias, factory=factory
            )
            for mask in masks
        )
        # Three dimensions keep channels_last conversion from rewriting strides.
        identity = torch.zeros(co, ci, 1, **factory)
        if ci == co:
            identity[:, :, 0].copy_(torch.eye(ci, **factory))
        self.register_buffer("identity_weight", identity, persistent=False)
        with torch.no_grad():
            # An exact identity route carries the original dense kernel. Other
            # factors stay random where their zero readout initially gates them:
            # this avoids tying duplicated channels or a permanently dead pair.
            self.direct.expand.weight[:ci, :, 0, 0].copy_(torch.eye(ci, **factory))
            self.direct.spatial.weight[:co].zero_()
            self.direct.spatial.weight[:co, :ci].copy_(original.weight)
            self.direct.reduce.weight.zero_()
            self.direct.reduce.weight[:, :co, 0, 0].copy_(torch.eye(co, **factory))
            # The independent sequential branch and edge branches start at zero.
            self.sequential.spatial.weight.zero_()
            # Cancel identity through an exact pointwise route, adding these
            # terms before the dense kernel to preserve exactly-zero readouts.
            self.pointwise.expand.weight[:ci, :, 0, 0].copy_(torch.eye(ci, **factory))
            self.pointwise.reduce.weight.zero_()
            self.pointwise.reduce.weight[:, :ci, 0, 0].copy_(-identity[:, :, 0])
            if bias:
                for module in self.modules():
                    if isinstance(module, nn.Conv2d) and module.bias is not None:
                        module.bias.zero_()
                self.direct.reduce.bias.copy_(original.bias)
        self.fold_active = False
        self.register_buffer("fold_kernel", None, persistent=False)
        self.register_buffer("fold_bias", None, persistent=False)

    def equivalent_kernel(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        """The kernel and bias to convolve with: the clip's while one is active."""
        if self.fold_active and dtype == torch.float32:
            assert self.fold_kernel is not None
            return self.fold_kernel, self.fold_bias
        return self.compose(dtype)

    def compose(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
        """Compose EVERY branch using ordinary autograd, never feature maps."""
        kernel, bias = self.direct.equivalent_kernel(dtype)
        point, point_bias = self.pointwise.equivalent_kernel(dtype)
        point = point + self.identity_weight.to(dtype).unsqueeze(-1)
        kernel = kernel + F.pad(point, (1, 1, 1, 1))
        if bias is not None:
            bias = bias + point_bias
        for branch in (self.sequential, *self.edges):
            branch_kernel, branch_bias = branch.equivalent_kernel(dtype)
            kernel = kernel + branch_kernel
            if bias is not None:
                bias = bias + branch_bias
        return kernel, bias

    def forward(self, x: Tensor) -> Tensor:
        parameter_dtype = self.direct.expand.weight.dtype
        fold_dtype = (
            torch.float64 if parameter_dtype == torch.float64 else torch.float32
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            weight, bias = self.equivalent_kernel(fold_dtype)
        return F.conv2d(
            x,
            weight.to(parameter_dtype),
            None if bias is None else bias.to(parameter_dtype),
            padding=1,
        )

    def store_fold(self, name: str, value: Tensor, track: bool) -> Tensor:
        """Refresh the named buffer in place, recreating an unusable one.

        A tracked buffer gets a zero gradient of its own, so backward calls
        accumulate into ordinary memory in place.
        """
        leaf = getattr(self, name)
        if (
            leaf is None
            or not leaf.is_leaf
            or leaf.device != value.device
            or leaf.dtype != value.dtype
            or leaf.shape != value.shape
            or leaf.stride() != value.stride()
        ):
            leaf = value.detach().clone()
            setattr(self, name, leaf)
        else:
            with torch.no_grad():
                leaf.copy_(value)
        leaf.grad = torch.zeros_like(leaf) if track else None
        return leaf.requires_grad_(track)

    @torch.no_grad()
    def folded(self) -> nn.Conv2d:
        parameter = self.direct.expand.weight
        module = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            padding=1,
            bias=self.direct.reduce.bias is not None,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            weight, bias = self.compose(torch.float64)
        module.weight.copy_(weight)
        if bias is not None:
            module.bias.copy_(bias)
        return module.eval()


class DynamicRouter(nn.Module):
    """FP32/FP64 per-image GAP/LayerNorm/MLP routing, centred around 1/K."""

    def __init__(
        self,
        channels: int,
        experts: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.channels, self.num_experts = channels, experts
        self.fc1 = nn.Linear(channels, channels // 4, device=device, dtype=dtype)
        self.fc2 = nn.Linear(channels // 4, experts, device=device, dtype=dtype)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        dtype = (
            torch.float64 if self.fc1.weight.dtype == torch.float64 else torch.float32
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            pooled = x.mean((2, 3), dtype=dtype)
            normalized = F.layer_norm(pooled, (self.channels,), eps=1e-5)
            hidden = F.gelu(
                F.linear(normalized, self.fc1.weight.to(dtype), self.fc1.bias.to(dtype))
            )
            logits = F.linear(
                hidden, self.fc2.weight.to(dtype), self.fc2.bias.to(dtype)
            )
            return (logits.softmax(-1) - 1.0 / self.num_experts)[:, :, None, None]


def _premixed_conv3x3(
    x: Tensor, weight0: Tensor, bias0: Tensor, experts: Tensor, coefficients: Tensor
) -> Tensor:
    """One differentiable mixed kernel per sample, including batch > 1."""
    batch, channels, height, width = x.shape
    dtype = torch.float64 if experts.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        residual = coefficients.reshape(batch, experts.shape[0]).to(dtype) @ (
            experts.reshape(experts.shape[0], -1).to(dtype)
        )
        weight = (weight0.to(dtype).reshape(1, -1) + residual).reshape(
            batch * channels, channels, 3, 3
        )
        bias = bias0.to(dtype).repeat(batch)
    weight, bias = weight.to(experts.dtype), bias.to(experts.dtype)
    if batch == 1:
        return F.conv2d(x, weight, bias, padding=1)
    output = F.conv2d(
        x.reshape(1, batch * channels, height, width),
        weight,
        bias,
        padding=1,
        groups=batch,
    ).reshape(batch, channels, height, width)
    if x.is_contiguous(memory_format=torch.channels_last):
        output = output.contiguous(memory_format=torch.channels_last)
    return output


class PremixedDynamicGroupConv3x3(nn.Module):
    """EDBB base plus K zero-initialized, input-routed residual kernels."""

    def __init__(self, base: EDBBReparamConv3x3, experts: int) -> None:
        super().__init__()
        self.base = base
        parameter = base.direct.expand.weight
        factory = {"device": parameter.device, "dtype": parameter.dtype}
        channels = base.in_channels
        self.experts = nn.Parameter(
            torch.zeros(experts, channels, channels * 9, **factory)
        )
        self.router = DynamicRouter(channels, experts, **factory)
        self.train(base.training)

    def coefficients(self, x: Tensor) -> Tensor:
        return self.router(x)

    def forward(self, x: Tensor) -> Tensor:
        dtype = torch.float64 if self.experts.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            weight, bias = self.base.equivalent_kernel(dtype)
        return _premixed_conv3x3(x, weight, bias, self.experts, self.coefficients(x))

    @torch.no_grad()
    def folded(self) -> PremixedDeployedDynamicConv3x3:
        return PremixedDeployedDynamicConv3x3(
            self.base.folded(), self.experts, self.router
        ).eval()


class PremixedDeployedDynamicConv3x3(nn.Module):
    """Static base folded; routing is recomputed for every input."""

    def __init__(self, base: nn.Conv2d, experts: Tensor, router: DynamicRouter) -> None:
        super().__init__()
        self.base = base
        self.experts = nn.Parameter(experts.detach().clone())
        self.router = deepcopy(router)

    def coefficients(self, x: Tensor) -> Tensor:
        return self.router(x)

    def forward(self, x: Tensor) -> Tensor:
        return _premixed_conv3x3(
            x, self.base.weight, self.base.bias, self.experts, self.coefficients(x)
        )


# Residual layer and group.


class _Layer(nn.Module):
    """Pre-normalized spatial attention and FFN, each with a residual addition."""

    def __init__(
        self, channels: int, window: int, *, shifted: bool, context: bool
    ) -> None:
        super().__init__()
        # Construction order is fixed to keep seeded initialization reproducible.
        self.norm1 = TokenNorm(channels, 1e-5)
        self.norm2 = TokenNorm(channels, 1e-5)
        self.attention = _SpatialAttention(
            channels, channels // 32, window, shifted=shifted, context=context
        )
        self.ffn = ConvFFN(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _Group(nn.Module):
    """The six-slot local/context schedule followed by a group write-back.

    Slots are [local, shifted local, context-small, local, shifted local,
    context-large] with G16 local windows. RDUAL wraps the context slots in
    DualContextLayer and then keeps only its selected slots.
    """

    def __init__(
        self, channels: int, c_windows: tuple[int, int], use_checkpoint: bool
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        schedule = (
            (16, False, False),
            (16, True, False),
            (c_windows[0], False, True),
            (16, False, False),
            (16, True, False),
            (c_windows[1], False, True),
        )
        self.layers = nn.ModuleList(
            _Layer(channels, window, shifted=shifted, context=context)
            for window, shifted, context in schedule
        )
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for layer in self.layers:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = _checkpoint_layer(layer, x)
            else:
                x = layer(x)
        return residual + self.conv(x)


# Optional Triton kernels: the direct local-attention backend.
# The bundled dense-RPB kernels serve the G16 local windows, not context NA.
# Without Triton or the torch.library APIs every path runs in PyTorch.
_DIRECT_ATTENTION_ERROR: str | None = None
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as error:
    if error.name not in {"triton", "triton.language"}:
        raise
    _DIRECT_ATTENTION_ERROR = f"missing optional dependency: {error.name}"
else:
    _missing_torch_apis = [
        name
        for name in ("triton_op", "wrap_triton", "register_autograd")
        if not callable(getattr(torch.library, name, None))
    ]
    if _missing_torch_apis:
        _DIRECT_ATTENTION_ERROR = "missing torch.library API(s): " + ", ".join(
            _missing_torch_apis
        )
    else:
        from torch.library import triton_op, wrap_triton

_HAS_DIRECT_ATTENTION = _DIRECT_ATTENTION_ERROR is None

if _HAS_DIRECT_ATTENTION:
    _BF16 = 1
    _FP16 = 2

    _FWD_BLOCK_M = 64
    _FWD_BLOCK_N = 64
    _FWD_WARPS = 4
    _FWD_STAGES = 3
    _DQ_BLOCK_M = 16
    _DQ_BLOCK_N = 64
    _DQ_WARPS = 4
    _DQ_STAGES = 3
    _DKV_BLOCK_M = 64
    _DKV_BLOCK_N = 64
    _DKV_WARPS = 4
    _DKV_STAGES = 3
    _DBIAS_BLOCK_M = 32
    _DBIAS_BLOCK_N = 64
    _DBIAS_WARPS = 4
    _DBIAS_STAGES = 3
    _DELTA_WARPS = 8
    _REDUCE_WARPS = 8

    def _attention_dtype_code(dtype: torch.dtype) -> int:
        """The kernels' BF16/FP16 switch; callers pass one of the two."""
        return _BF16 if dtype is torch.bfloat16 else _FP16

    @triton.jit
    def _rounded_bias(value, DTYPE_CODE: tl.constexpr):
        if DTYPE_CODE == 1:
            return value.to(tl.bfloat16).to(tl.float32)
        return value.to(tl.float16).to(tl.float32)

    @triton.jit
    def _attention_forward_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        output_ptr,
        lse_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_token,
        query_stride_channel,
        key_stride_batch,
        key_stride_head,
        key_stride_token,
        key_stride_channel,
        value_stride_batch,
        value_stride_head,
        value_stride_token,
        value_stride_channel,
        output_stride_batch,
        output_stride_head,
        output_stride_token,
        output_stride_channel,
        table_stride_relative,
        table_stride_head,
        scale,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_width: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads

        query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        channel = tl.arange(0, BLOCK_D)
        query_valid = query_index < query_tokens
        channel_valid = channel < head_dim

        query_offsets = (
            batch * query_stride_batch
            + head * query_stride_head
            + query_index[:, None] * query_stride_token
            + channel[None, :] * query_stride_channel
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        relative_width = query_width + key_width - 1

        for key_start in range(0, key_tokens, BLOCK_N):
            key_index = key_start + tl.arange(0, BLOCK_N)
            key_valid = key_index < key_tokens
            key_offsets = (
                batch * key_stride_batch
                + head * key_stride_head
                + key_index[:, None] * key_stride_token
                + channel[None, :] * key_stride_channel
            )
            key = tl.load(
                key_ptr + key_offsets,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)

            key_y = key_index // key_width
            key_x = key_index - key_y * key_width
            relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
            relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
            relative_index = relative_y * relative_width + relative_x
            bias = tl.load(
                table_ptr
                + relative_index * table_stride_relative
                + head * table_stride_head,
                mask=query_valid[:, None] & key_valid[None, :],
                other=0.0,
            )
            score += _rounded_bias(bias, DTYPE_CODE)
            score = tl.where(
                query_valid[:, None] & key_valid[None, :],
                score,
                -float("inf"),
            )

            block_max = tl.max(score, axis=1)
            new_max = tl.maximum(row_max, block_max)
            probability = tl.exp(score - new_max[:, None])
            correction = tl.exp(row_max - new_max)
            block_sum = tl.sum(probability, axis=1)
            row_sum = row_sum * correction + block_sum
            accumulator *= correction[:, None]

            value_offsets = (
                batch * value_stride_batch
                + head * value_stride_head
                + key_index[:, None] * value_stride_token
                + channel[None, :] * value_stride_channel
            )
            value = tl.load(
                value_ptr + value_offsets,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            if DTYPE_CODE == 1:
                accumulator += tl.dot(probability.to(tl.bfloat16), value)
            else:
                accumulator += tl.dot(probability.to(tl.float16), value)
            row_max = new_max

        accumulator /= row_sum[:, None]
        output_offsets = (
            batch * output_stride_batch
            + head * output_stride_head
            + query_index[:, None] * output_stride_token
            + channel[None, :] * output_stride_channel
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator,
            mask=query_valid[:, None] & channel_valid[None, :],
        )
        lse_offset = (batch * num_heads + head) * query_tokens + query_index
        tl.store(
            lse_ptr + lse_offset,
            row_max + tl.log(row_sum),
            mask=query_valid,
        )

    @triton.jit
    def _attention_delta_kernel(
        output_ptr,
        grad_output_ptr,
        delta_ptr,
        output_stride_batch,
        output_stride_head,
        output_stride_token,
        output_stride_channel,
        grad_stride_batch,
        grad_stride_head,
        grad_stride_token,
        grad_stride_channel,
        query_tokens: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_index = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads
        channel = tl.arange(0, BLOCK_D)
        valid = channel < head_dim
        output_offset = (
            batch * output_stride_batch
            + head * output_stride_head
            + query_index * output_stride_token
            + channel * output_stride_channel
        )
        grad_offset = (
            batch * grad_stride_batch
            + head * grad_stride_head
            + query_index * grad_stride_token
            + channel * grad_stride_channel
        )
        output = tl.load(output_ptr + output_offset, mask=valid, other=0.0).to(
            tl.float32
        )
        grad_output = tl.load(
            grad_output_ptr + grad_offset,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(output * grad_output, axis=0)
        tl.store(delta_ptr + batch_head * query_tokens + query_index, delta)

    @triton.jit
    def _attention_backward_query_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        grad_query_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_token,
        query_stride_channel,
        key_stride_batch,
        key_stride_head,
        key_stride_token,
        key_stride_channel,
        value_stride_batch,
        value_stride_head,
        value_stride_token,
        value_stride_channel,
        grad_output_stride_batch,
        grad_output_stride_head,
        grad_output_stride_token,
        grad_output_stride_channel,
        grad_query_stride_batch,
        grad_query_stride_head,
        grad_query_stride_token,
        grad_query_stride_channel,
        table_stride_relative,
        table_stride_head,
        scale,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_width: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads
        query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        channel = tl.arange(0, BLOCK_D)
        query_valid = query_index < query_tokens
        channel_valid = channel < head_dim

        query_offset = (
            batch * query_stride_batch
            + head * query_stride_head
            + query_index[:, None] * query_stride_token
            + channel[None, :] * query_stride_channel
        )
        query = tl.load(
            query_ptr + query_offset,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        grad_output_offset = (
            batch * grad_output_stride_batch
            + head * grad_output_stride_head
            + query_index[:, None] * grad_output_stride_token
            + channel[None, :] * grad_output_stride_channel
        )
        grad_output = tl.load(
            grad_output_ptr + grad_output_offset,
            mask=query_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + batch_head * query_tokens + query_index,
            mask=query_valid,
            other=0.0,
        )
        delta = tl.load(
            delta_ptr + batch_head * query_tokens + query_index,
            mask=query_valid,
            other=0.0,
        )
        grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        relative_width = query_width + key_width - 1

        for key_start in range(0, key_tokens, BLOCK_N):
            key_index = key_start + tl.arange(0, BLOCK_N)
            key_valid = key_index < key_tokens
            key_offset = (
                batch * key_stride_batch
                + head * key_stride_head
                + key_index[:, None] * key_stride_token
                + channel[None, :] * key_stride_channel
            )
            key = tl.load(
                key_ptr + key_offset,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            value_offset = (
                batch * value_stride_batch
                + head * value_stride_head
                + key_index[:, None] * value_stride_token
                + channel[None, :] * value_stride_channel
            )
            value = tl.load(
                value_ptr + value_offset,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )

            score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)
            key_y = key_index // key_width
            key_x = key_index - key_y * key_width
            relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
            relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
            relative_index = relative_y * relative_width + relative_x
            bias = tl.load(
                table_ptr
                + relative_index * table_stride_relative
                + head * table_stride_head,
                mask=query_valid[:, None] & key_valid[None, :],
                other=0.0,
            )
            score += _rounded_bias(bias, DTYPE_CODE)
            probability = tl.exp(score - lse[:, None])
            probability = tl.where(
                query_valid[:, None] & key_valid[None, :],
                probability,
                0.0,
            )
            grad_probability = tl.dot(grad_output, tl.trans(value))
            grad_score = probability * (grad_probability - delta[:, None])
            if DTYPE_CODE == 1:
                grad_query += tl.dot(grad_score.to(tl.bfloat16), key) * scale.to(
                    tl.float32
                )
            else:
                grad_query += tl.dot(grad_score.to(tl.float16), key) * scale.to(
                    tl.float32
                )

        grad_query_offset = (
            batch * grad_query_stride_batch
            + head * grad_query_stride_head
            + query_index[:, None] * grad_query_stride_token
            + channel[None, :] * grad_query_stride_channel
        )
        tl.store(
            grad_query_ptr + grad_query_offset,
            grad_query,
            mask=query_valid[:, None] & channel_valid[None, :],
        )

    @triton.jit
    def _attention_backward_key_value_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        grad_key_ptr,
        grad_value_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_token,
        query_stride_channel,
        key_stride_batch,
        key_stride_head,
        key_stride_token,
        key_stride_channel,
        value_stride_batch,
        value_stride_head,
        value_stride_token,
        value_stride_channel,
        grad_output_stride_batch,
        grad_output_stride_head,
        grad_output_stride_token,
        grad_output_stride_channel,
        grad_key_stride_batch,
        grad_key_stride_head,
        grad_key_stride_token,
        grad_key_stride_channel,
        grad_value_stride_batch,
        grad_value_stride_head,
        grad_value_stride_token,
        grad_value_stride_channel,
        table_stride_relative,
        table_stride_head,
        scale,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_width: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // num_heads
        head = batch_head - batch * num_heads
        key_index = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        channel = tl.arange(0, BLOCK_D)
        key_valid = key_index < key_tokens
        channel_valid = channel < head_dim

        key_offset = (
            batch * key_stride_batch
            + head * key_stride_head
            + key_index[:, None] * key_stride_token
            + channel[None, :] * key_stride_channel
        )
        key = tl.load(
            key_ptr + key_offset,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        value_offset = (
            batch * value_stride_batch
            + head * value_stride_head
            + key_index[:, None] * value_stride_token
            + channel[None, :] * value_stride_channel
        )
        value = tl.load(
            value_ptr + value_offset,
            mask=key_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        grad_key = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        grad_value = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        key_y = key_index // key_width
        key_x = key_index - key_y * key_width
        relative_width = query_width + key_width - 1

        for query_start in range(0, query_tokens, BLOCK_M):
            query_index = query_start + tl.arange(0, BLOCK_M)
            query_valid = query_index < query_tokens
            query_offset = (
                batch * query_stride_batch
                + head * query_stride_head
                + query_index[:, None] * query_stride_token
                + channel[None, :] * query_stride_channel
            )
            query = tl.load(
                query_ptr + query_offset,
                mask=query_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            grad_output_offset = (
                batch * grad_output_stride_batch
                + head * grad_output_stride_head
                + query_index[:, None] * grad_output_stride_token
                + channel[None, :] * grad_output_stride_channel
            )
            grad_output = tl.load(
                grad_output_ptr + grad_output_offset,
                mask=query_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr + batch_head * query_tokens + query_index,
                mask=query_valid,
                other=0.0,
            )
            delta = tl.load(
                delta_ptr + batch_head * query_tokens + query_index,
                mask=query_valid,
                other=0.0,
            )

            score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32)
            query_y = query_index // query_width
            query_x = query_index - query_y * query_width
            relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
            relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
            relative_index = relative_y * relative_width + relative_x
            bias = tl.load(
                table_ptr
                + relative_index * table_stride_relative
                + head * table_stride_head,
                mask=query_valid[:, None] & key_valid[None, :],
                other=0.0,
            )
            score += _rounded_bias(bias, DTYPE_CODE)
            probability = tl.exp(score - lse[:, None])
            probability = tl.where(
                query_valid[:, None] & key_valid[None, :],
                probability,
                0.0,
            )
            grad_probability = tl.dot(grad_output, tl.trans(value))
            grad_score = probability * (grad_probability - delta[:, None])
            if DTYPE_CODE == 1:
                grad_key += tl.dot(
                    tl.trans(grad_score.to(tl.bfloat16)), query
                ) * scale.to(tl.float32)
                grad_value += tl.dot(tl.trans(probability.to(tl.bfloat16)), grad_output)
            else:
                grad_key += tl.dot(
                    tl.trans(grad_score.to(tl.float16)), query
                ) * scale.to(tl.float32)
                grad_value += tl.dot(tl.trans(probability.to(tl.float16)), grad_output)

        grad_key_offset = (
            batch * grad_key_stride_batch
            + head * grad_key_stride_head
            + key_index[:, None] * grad_key_stride_token
            + channel[None, :] * grad_key_stride_channel
        )
        grad_value_offset = (
            batch * grad_value_stride_batch
            + head * grad_value_stride_head
            + key_index[:, None] * grad_value_stride_token
            + channel[None, :] * grad_value_stride_channel
        )
        mask = key_valid[:, None] & channel_valid[None, :]
        tl.store(grad_key_ptr + grad_key_offset, grad_key, mask=mask)
        tl.store(grad_value_ptr + grad_value_offset, grad_value, mask=mask)

    @triton.jit
    def _attention_backward_dense_bias_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        table_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        grad_bias_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_token,
        query_stride_channel,
        key_stride_batch,
        key_stride_head,
        key_stride_token,
        key_stride_channel,
        value_stride_batch,
        value_stride_head,
        value_stride_token,
        value_stride_channel,
        grad_output_stride_batch,
        grad_output_stride_head,
        grad_output_stride_token,
        grad_output_stride_channel,
        table_stride_relative,
        table_stride_head,
        scale,
        batch_windows: tl.constexpr,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_width: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        key_block = tl.program_id(1)
        head = tl.program_id(2)
        query_index = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_index = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        channel = tl.arange(0, BLOCK_D)
        query_valid = query_index < query_tokens
        key_valid = key_index < key_tokens
        channel_valid = channel < head_dim
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        key_y = key_index // key_width
        key_x = key_index - key_y * key_width
        relative_width = query_width + key_width - 1
        relative_y = key_y[None, :] - query_y[:, None] + query_height - 1
        relative_x = key_x[None, :] - query_x[:, None] + query_width - 1
        relative_index = relative_y * relative_width + relative_x
        bias = tl.load(
            table_ptr
            + relative_index * table_stride_relative
            + head * table_stride_head,
            mask=query_valid[:, None] & key_valid[None, :],
            other=0.0,
        )
        bias = _rounded_bias(bias, DTYPE_CODE)
        grad_bias = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for batch in range(batch_windows):
            query_offset = (
                batch * query_stride_batch
                + head * query_stride_head
                + query_index[:, None] * query_stride_token
                + channel[None, :] * query_stride_channel
            )
            key_offset = (
                batch * key_stride_batch
                + head * key_stride_head
                + key_index[:, None] * key_stride_token
                + channel[None, :] * key_stride_channel
            )
            value_offset = (
                batch * value_stride_batch
                + head * value_stride_head
                + key_index[:, None] * value_stride_token
                + channel[None, :] * value_stride_channel
            )
            grad_output_offset = (
                batch * grad_output_stride_batch
                + head * grad_output_stride_head
                + query_index[:, None] * grad_output_stride_token
                + channel[None, :] * grad_output_stride_channel
            )
            query = tl.load(
                query_ptr + query_offset,
                mask=query_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            key = tl.load(
                key_ptr + key_offset,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            value = tl.load(
                value_ptr + value_offset,
                mask=key_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            grad_output = tl.load(
                grad_output_ptr + grad_output_offset,
                mask=query_valid[:, None] & channel_valid[None, :],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr + (batch * num_heads + head) * query_tokens + query_index,
                mask=query_valid,
                other=0.0,
            )
            row_delta = tl.load(
                delta_ptr + (batch * num_heads + head) * query_tokens + query_index,
                mask=query_valid,
                other=0.0,
            )
            score = tl.dot(query, tl.trans(key)) * scale.to(tl.float32) + bias
            probability = tl.exp(score - lse[:, None])
            probability = tl.where(
                query_valid[:, None] & key_valid[None, :],
                probability,
                0.0,
            )
            grad_probability = tl.dot(grad_output, tl.trans(value))
            grad_bias += probability * (grad_probability - row_delta[:, None])

        output_offset = (
            head * query_tokens + query_index[:, None]
        ) * key_tokens + key_index[None, :]
        tl.store(
            grad_bias_ptr + output_offset,
            grad_bias,
            mask=query_valid[:, None] & key_valid[None, :],
        )

    @triton.jit
    def _reduce_dense_bias_gradient_kernel(
        grad_bias_ptr,
        grad_table_ptr,
        query_tokens: tl.constexpr,
        key_tokens: tl.constexpr,
        query_height: tl.constexpr,
        query_width: tl.constexpr,
        key_height: tl.constexpr,
        key_width: tl.constexpr,
        relative_width: tl.constexpr,
        num_heads: tl.constexpr,
        BLOCK_QUERY: tl.constexpr,
    ):
        output_index = tl.program_id(0)
        head = output_index % num_heads
        relative_index = output_index // num_heads
        relative_y = relative_index // relative_width
        relative_x = relative_index - relative_y * relative_width
        delta_y = relative_y - (query_height - 1)
        delta_x = relative_x - (query_width - 1)

        query_index = tl.arange(0, BLOCK_QUERY)
        query_y = query_index // query_width
        query_x = query_index - query_y * query_width
        key_y = query_y + delta_y
        key_x = query_x + delta_x
        valid = (
            (query_index < query_tokens)
            & (key_y >= 0)
            & (key_y < key_height)
            & (key_x >= 0)
            & (key_x < key_width)
        )
        key_index = key_y * key_width + key_x
        dense_offset = (head * query_tokens + query_index) * key_tokens + key_index
        gradient = tl.load(
            grad_bias_ptr + dense_offset,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_table_ptr + relative_index * num_heads + head,
            tl.sum(gradient, axis=0),
        )

    @triton_op("rdual_standalone::dense_rpb_attention", mutates_args={})
    def _dense_rpb_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        query_height: int,
        query_width: int,
        key_height: int,
        key_width: int,
        scale: float,
        dtype_code: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_windows, num_heads, query_tokens, head_dim = query.shape
        # A contiguous head-major result; the kernels take explicit strides.
        output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
        lse = torch.empty(
            (batch_windows, num_heads, query_tokens),
            device=query.device,
            dtype=torch.float32,
        )
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (
            triton.cdiv(query_tokens, _FWD_BLOCK_M),
            batch_windows * num_heads,
        )
        wrap_triton(_attention_forward_kernel)[grid](
            query,
            key,
            value,
            table,
            output,
            lse,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            *table.stride(),
            scale,
            query_tokens,
            key_height * key_width,
            query_height,
            query_width,
            key_width,
            num_heads,
            head_dim,
            dtype_code,
            BLOCK_M=_FWD_BLOCK_M,
            BLOCK_N=_FWD_BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=_FWD_WARPS,
            num_stages=_FWD_STAGES,
        )
        return output, lse

    @triton_op("rdual_standalone::dense_rpb_attention_backward", mutates_args={})
    def _dense_rpb_attention_backward(
        grad_output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        query_height: int,
        query_width: int,
        key_height: int,
        key_width: int,
        scale: float,
        dtype_code: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_output = grad_output.contiguous()
        batch_windows, num_heads, query_tokens, head_dim = query.shape
        key_tokens = key_height * key_width
        BLOCK_D = triton.next_power_of_2(head_dim)
        delta = torch.empty_like(lse)
        delta_grid = (query_tokens, batch_windows * num_heads)
        wrap_triton(_attention_delta_kernel)[delta_grid](
            output,
            grad_output,
            delta,
            *output.stride(),
            *grad_output.stride(),
            query_tokens,
            num_heads,
            head_dim,
            BLOCK_D=BLOCK_D,
            num_warps=_DELTA_WARPS,
        )

        # Layout-preserving gradients avoid copies when restoring QKV storage.
        # All kernels accept explicit source/destination strides.
        grad_query = torch.empty(query.shape, device=query.device, dtype=query.dtype)
        query_grid = (
            triton.cdiv(query_tokens, _DQ_BLOCK_M),
            batch_windows * num_heads,
        )
        wrap_triton(_attention_backward_query_kernel)[query_grid](
            query,
            key,
            value,
            table,
            grad_output,
            lse,
            delta,
            grad_query,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *grad_query.stride(),
            *table.stride(),
            scale,
            query_tokens,
            key_tokens,
            query_height,
            query_width,
            key_width,
            num_heads,
            head_dim,
            dtype_code,
            BLOCK_M=_DQ_BLOCK_M,
            BLOCK_N=_DQ_BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=_DQ_WARPS,
            num_stages=_DQ_STAGES,
        )

        grad_key = torch.empty(key.shape, device=key.device, dtype=key.dtype)
        grad_value = torch.empty(value.shape, device=value.device, dtype=value.dtype)
        key_value_grid = (
            triton.cdiv(key_tokens, _DKV_BLOCK_N),
            batch_windows * num_heads,
        )
        wrap_triton(_attention_backward_key_value_kernel)[key_value_grid](
            query,
            key,
            value,
            table,
            grad_output,
            lse,
            delta,
            grad_key,
            grad_value,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *grad_key.stride(),
            *grad_value.stride(),
            *table.stride(),
            scale,
            query_tokens,
            key_tokens,
            query_height,
            query_width,
            key_width,
            num_heads,
            head_dim,
            dtype_code,
            BLOCK_M=_DKV_BLOCK_M,
            BLOCK_N=_DKV_BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=_DKV_WARPS,
            num_stages=_DKV_STAGES,
        )

        grad_bias = torch.empty(
            (num_heads, query_tokens, key_tokens),
            device=query.device,
            dtype=query.dtype,
        )
        dense_bias_grid = (
            triton.cdiv(query_tokens, _DBIAS_BLOCK_M),
            triton.cdiv(key_tokens, _DBIAS_BLOCK_N),
            num_heads,
        )
        wrap_triton(_attention_backward_dense_bias_kernel)[dense_bias_grid](
            query,
            key,
            value,
            table,
            grad_output,
            lse,
            delta,
            grad_bias,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *table.stride(),
            scale,
            batch_windows,
            query_tokens,
            key_tokens,
            query_height,
            query_width,
            key_width,
            num_heads,
            head_dim,
            dtype_code,
            BLOCK_M=_DBIAS_BLOCK_M,
            BLOCK_N=_DBIAS_BLOCK_N,
            BLOCK_D=BLOCK_D,
            num_warps=_DBIAS_WARPS,
            num_stages=_DBIAS_STAGES,
        )

        grad_table = torch.empty_like(table, memory_format=torch.contiguous_format)
        relative_entries = table.shape[0]
        relative_width = query_width + key_width - 1
        reduction_grid = (relative_entries * num_heads,)
        wrap_triton(_reduce_dense_bias_gradient_kernel)[reduction_grid](
            grad_bias,
            grad_table,
            query_tokens,
            key_tokens,
            query_height,
            query_width,
            key_height,
            key_width,
            relative_width,
            num_heads,
            BLOCK_QUERY=triton.next_power_of_2(query_tokens),
            num_warps=_REDUCE_WARPS,
        )
        return grad_query, grad_key, grad_value, grad_table

    def _setup_context(ctx, inputs, output) -> None:
        (
            query,
            key,
            value,
            table,
            query_height,
            query_width,
            key_height,
            key_width,
            scale,
            dtype_code,
        ) = inputs
        attention_output, lse = output
        ctx.save_for_backward(query, key, value, table, attention_output, lse)
        ctx.query_height = query_height
        ctx.query_width = query_width
        ctx.key_height = key_height
        ctx.key_width = key_width
        ctx.scale = scale
        ctx.dtype_code = dtype_code

    def _backward(ctx, grad_output: torch.Tensor, grad_lse: torch.Tensor | None):
        del grad_lse
        query, key, value, table, attention_output, lse = ctx.saved_tensors
        grad_query, grad_key, grad_value, grad_table = _dense_rpb_attention_backward(
            grad_output,
            query,
            key,
            value,
            table,
            attention_output,
            lse,
            ctx.query_height,
            ctx.query_width,
            ctx.key_height,
            ctx.key_width,
            ctx.scale,
            ctx.dtype_code,
        )
        return (
            grad_query,
            grad_key,
            grad_value,
            grad_table,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        _dense_rpb_attention,
        _backward,
        setup_context=_setup_context,
    )

    def direct_attention(
        query: Tensor, key: Tensor, value: Tensor, table: Tensor, window: int
    ) -> Tensor:
        """Dense-RPB attention over complete window x window local windows.

        Q is already scaled. Model tables use query-minus-key; the kernel uses
        key-minus-query. Reversing the flattened table converts between them.
        """
        output, _lse = _dense_rpb_attention(
            query,
            key,
            value,
            table.flip(0),
            window,
            window,
            window,
            window,
            1.0,
            _attention_dtype_code(query.dtype),
        )
        return output


# Optional temporal CUDA kernels, guarded by the same optional APIs.
# Streaming temporal attention retains the full query-centered donor set.
# Probability/value products and saved aggregates use FP32 (TF32x3 dot), not a
# BF16 probability tensor. Shared K/V and positional gradients accumulate in
# FP32 atomics; deterministic execution uses the PyTorch reference instead.
# The joint K/V warp packs projected keys and values and samples them with a
# frozen-flow bilinear border warp; only feature adjoints are defined.
if _HAS_DIRECT_ATTENTION:

    @triton.jit
    def _temporal_forward_kernel(
        Q: tl.tensor,
        K: tl.tensor,
        V: tl.tensor,
        R: tl.tensor,
        NK: tl.tensor,
        NB: tl.tensor,
        SB: tl.tensor,
        Valid: tl.tensor,
        Output: tl.tensor,
        LSE: tl.tensor,
        H: tl.constexpr,
        W: tl.constexpr,
        S: tl.constexpr,
        RADIUS: tl.constexpr,
        UY: tl.constexpr,
        UX: tl.constexpr,
        TILE: tl.constexpr,
        BN: tl.constexpr,
    ) -> None:
        tile, bh = tl.program_id(0), tl.program_id(1)
        b, head = bh // 4, bh % 4
        nx: tl.constexpr = triton.cdiv(W, TILE)
        m = tl.arange(0, TILE * TILE)
        d = tl.arange(0, 32)
        py, px = tile // nx * TILE + m // TILE, tile % nx * TILE + m % TILE
        pos = py * W + px
        qm = (py < H) & (px < W)
        q = tl.load(Q + (bh * H * W + pos[:, None]) * 32 + d[None, :], qm[:, None], 0)
        nk = tl.load(NK + head * 32 + d)
        null_score = tl.sum(q.to(tl.float32) * nk[None, :].to(tl.float32), 1) * (
            32**-0.5
        ) + tl.load(NB + head)
        maximum = null_score
        denom = tl.full((TILE * TILE,), 1.0, tl.float32)
        acc = tl.full((TILE * TILE, 32), 0.0, tl.float32)
        start_y = tl.minimum(tl.maximum(py - RADIUS, 0), H - min(H, 2 * RADIUS + 1))
        start_x = tl.minimum(tl.maximum(px - RADIUS, 0), W - min(W, 2 * RADIUS + 1))
        union_y = tl.minimum(tl.maximum(tile // nx * TILE - RADIUS, 0), H - UY)
        union_x = tl.minimum(tl.maximum(tile % nx * TILE - RADIUS, 0), W - UX)
        for first in range(triton.cdiv(S * UY * UX, BN)):
            n = first * BN + tl.arange(0, BN)
            slot = n // (UY * UX)
            ky = union_y + (n % (UY * UX)) // UX
            kx = union_x + n % UX
            keypos = ky * W + kx
            nm = n < S * UY * UX
            allowed = tl.load(Valid + b * S + slot, nm, 0) != 0
            index = ((b * S + slot) * 4 + head) * H * W + keypos
            k = tl.load(K + index[:, None] * 32 + d[None, :], nm[:, None], 0)
            v = tl.load(V + index[:, None] * 32 + d[None, :], nm[:, None], 0)
            support = (
                qm[:, None]
                & nm[None, :]
                & allowed[None, :]
                & (ky[None, :] >= start_y[:, None])
                & (ky[None, :] < start_y[:, None] + min(H, 2 * RADIUS + 1))
                & (kx[None, :] >= start_x[:, None])
                & (kx[None, :] < start_x[:, None] + min(W, 2 * RADIUS + 1))
            )
            ri = (
                (py[:, None] - ky[None, :] + 2 * RADIUS) * (4 * RADIUS + 1)
                + px[:, None]
                - kx[None, :]
                + 2 * RADIUS
            )
            rb = tl.load(R + head * (4 * RADIUS + 1) ** 2 + tl.where(support, ri, 0))
            sb = tl.load(SB + slot * 4 + head, nm, 0)
            score = (
                tl.dot(q, tl.trans(k), input_precision="tf32x3") * (32**-0.5)
                + rb
                + sb[None, :]
            )
            score = tl.where(support, score, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            alpha = tl.exp(maximum - new_max)
            p = tl.exp(score - new_max[:, None])
            acc = acc * alpha[:, None] + tl.dot(
                p, v.to(tl.float32), input_precision="tf32x3"
            )
            denom = denom * alpha + tl.sum(p, 1)
            maximum = new_max
        out = acc / denom[:, None]
        tl.store(
            Output + (bh * H * W + pos[:, None]) * 32 + d[None, :], out, qm[:, None]
        )
        tl.store(LSE + bh * H * W + pos, maximum + tl.log(denom), qm)

    @triton.jit
    def _temporal_backward_kernel(
        Q: tl.tensor,
        K: tl.tensor,
        V: tl.tensor,
        R: tl.tensor,
        NK: tl.tensor,
        NB: tl.tensor,
        SB: tl.tensor,
        Valid: tl.tensor,
        Output: tl.tensor,
        LSE: tl.tensor,
        G: tl.tensor,
        DQ: tl.tensor,
        DK: tl.tensor,
        DV: tl.tensor,
        DR: tl.tensor,
        DNK: tl.tensor,
        DNB: tl.tensor,
        DSB: tl.tensor,
        H: tl.constexpr,
        W: tl.constexpr,
        S: tl.constexpr,
        RADIUS: tl.constexpr,
        UY: tl.constexpr,
        UX: tl.constexpr,
        TILE: tl.constexpr,
        BN: tl.constexpr,
    ) -> None:
        tile, bh = tl.program_id(0), tl.program_id(1)
        b, head = bh // 4, bh % 4
        nx: tl.constexpr = triton.cdiv(W, TILE)
        m = tl.arange(0, TILE * TILE)
        d = tl.arange(0, 32)
        py, px = tile // nx * TILE + m // TILE, tile % nx * TILE + m % TILE
        pos = py * W + px
        qm = (py < H) & (px < W)
        ptr = (bh * H * W + pos[:, None]) * 32 + d[None, :]
        q = tl.load(Q + ptr, qm[:, None], 0)
        g = tl.load(G + ptr, qm[:, None], 0).to(tl.float32)
        out = tl.load(Output + ptr, qm[:, None], 0)
        lse = tl.load(LSE + bh * H * W + pos, qm, 0)
        delta = tl.sum(g * out, 1)
        nk = tl.load(NK + head * 32 + d).to(tl.float32)
        ns = tl.sum(q.to(tl.float32) * nk[None, :], 1) * (32**-0.5) + tl.load(NB + head)
        dn = tl.where(qm, -tl.exp(ns - lse) * delta, 0.0)
        dq = dn[:, None] * nk[None, :] * (32**-0.5)
        tl.atomic_add(
            DNK + head * 32 + d,
            tl.sum(dn[:, None] * q.to(tl.float32), 0) * (32**-0.5),
        )
        tl.atomic_add(DNB + head, tl.sum(dn, 0))
        start_y = tl.minimum(tl.maximum(py - RADIUS, 0), H - min(H, 2 * RADIUS + 1))
        start_x = tl.minimum(tl.maximum(px - RADIUS, 0), W - min(W, 2 * RADIUS + 1))
        union_y = tl.minimum(tl.maximum(tile // nx * TILE - RADIUS, 0), H - UY)
        union_x = tl.minimum(tl.maximum(tile % nx * TILE - RADIUS, 0), W - UX)
        for first in range(triton.cdiv(S * UY * UX, BN)):
            n = first * BN + tl.arange(0, BN)
            slot = n // (UY * UX)
            ky, kx = union_y + (n % (UY * UX)) // UX, union_x + n % UX
            keypos = ky * W + kx
            nm = n < S * UY * UX
            allowed = tl.load(Valid + b * S + slot, nm, 0) != 0
            index = ((b * S + slot) * 4 + head) * H * W + keypos
            k = tl.load(K + index[:, None] * 32 + d[None, :], nm[:, None], 0)
            v = tl.load(V + index[:, None] * 32 + d[None, :], nm[:, None], 0)
            support = (
                qm[:, None]
                & nm[None, :]
                & allowed[None, :]
                & (ky[None, :] >= start_y[:, None])
                & (ky[None, :] < start_y[:, None] + min(H, 2 * RADIUS + 1))
                & (kx[None, :] >= start_x[:, None])
                & (kx[None, :] < start_x[:, None] + min(W, 2 * RADIUS + 1))
            )
            ri = (
                (py[:, None] - ky[None, :] + 2 * RADIUS) * (4 * RADIUS + 1)
                + px[:, None]
                - kx[None, :]
                + 2 * RADIUS
            )
            rb = tl.load(R + head * (4 * RADIUS + 1) ** 2 + tl.where(support, ri, 0))
            sb = tl.load(SB + slot * 4 + head, nm, 0)
            score = (
                tl.dot(q, tl.trans(k), input_precision="tf32x3") * (32**-0.5)
                + rb
                + sb[None, :]
            )
            p = tl.where(support, tl.exp(score - lse[:, None]), 0.0)
            dp = tl.dot(g, tl.trans(v).to(tl.float32), input_precision="tf32x3")
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds, k.to(tl.float32), input_precision="tf32x3") * (32**-0.5)
            dk = tl.dot(tl.trans(ds), q.to(tl.float32), input_precision="tf32x3") * (
                32**-0.5
            )
            dv = tl.dot(tl.trans(p), g, input_precision="tf32x3")
            tl.atomic_add(DK + index[:, None] * 32 + d[None, :], dk, nm[:, None])
            tl.atomic_add(DV + index[:, None] * 32 + d[None, :], dv, nm[:, None])
            tl.atomic_add(
                DR + head * (4 * RADIUS + 1) ** 2 + tl.where(support, ri, 0),
                ds,
                support,
            )
            tl.atomic_add(DSB + slot * 4 + head, tl.sum(ds, 0), nm)
        tl.store(DQ + ptr, dq, qm[:, None])

    @triton_op("rdual_standalone::attention", mutates_args={})
    def _temporal_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        table: torch.Tensor,
        null_key: torch.Tensor,
        null_bias: torch.Tensor,
        slot_bias: torch.Tensor,
        valid: torch.Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(q, dtype=torch.float32)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        tile = 8
        wrap_triton(_temporal_forward_kernel)[
            (triton.cdiv(height, tile) * triton.cdiv(width, tile), q.shape[0] * 4)
        ](
            q,
            k,
            v,
            table,
            null_key,
            null_bias,
            slot_bias,
            valid,
            out,
            lse,
            height,
            width,
            k.shape[1],
            radius,
            min(height, tile + 2 * radius),
            min(width, tile + 2 * radius),
            tile,
            64,
            num_warps=4,
            num_stages=1,
        )
        return out, lse

    @triton_op("rdual_standalone::attention_backward", mutates_args={})
    def _temporal_attention_backward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        table: torch.Tensor,
        nk: torch.Tensor,
        nb: torch.Tensor,
        sb: torch.Tensor,
        valid: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        g: torch.Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> list[torch.Tensor]:
        dq = torch.empty_like(q, dtype=torch.float32)
        grads = [
            torch.zeros_like(t, dtype=torch.float32) for t in (k, v, table, nk, nb, sb)
        ]
        tile = 8
        wrap_triton(_temporal_backward_kernel)[
            (triton.cdiv(height, tile) * triton.cdiv(width, tile), q.shape[0] * 4)
        ](
            q,
            k,
            v,
            table,
            nk,
            nb,
            sb,
            valid,
            out,
            lse,
            g.contiguous(),
            dq,
            *grads,
            height,
            width,
            k.shape[1],
            radius,
            min(height, tile + 2 * radius),
            min(width, tile + 2 * radius),
            tile,
            64,
            num_warps=4,
            num_stages=1,
        )
        return [
            dq.to(q.dtype),
            *[
                grad.to(t.dtype)
                for grad, t in zip(grads, (k, v, table, nk, nb, sb), strict=True)
            ],
        ]

    def _temporal_setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        ctx.save_for_backward(*inputs[:8], *output)
        ctx.geometry = inputs[8:]
        ctx.mark_non_differentiable(output[1])

    def _temporal_backward(
        ctx: Any, grad_out: torch.Tensor, _grad_lse: torch.Tensor | None
    ) -> tuple[Any, ...]:
        return (
            *_temporal_attention_backward(*ctx.saved_tensors, grad_out, *ctx.geometry),
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        _temporal_attention,
        _temporal_backward,
        setup_context=_temporal_setup_context,
    )

    def temporal_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        table: torch.Tensor,
        null_key: torch.Tensor,
        null_bias: torch.Tensor,
        slot_bias: torch.Tensor,
        valid: torch.Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> torch.Tensor:
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "Temporal atomic backward requires nondeterministic mode; "
                "use reference backend for deterministic execution"
            )
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Temporal CUDA kernel supports FP16/BF16/FP32, not FP64")
        out, _lse = _temporal_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            table.contiguous(),
            null_key.to(q.dtype).contiguous(),
            null_bias.contiguous(),
            slot_bias.contiguous(),
            valid.contiguous(),
            height,
            width,
            radius,
        )
        return out.to(v.dtype)

    @triton.jit
    def _kv_warp_forward_kernel(
        X: tl.tensor,
        Flow: tl.tensor,
        Y: tl.tensor,
        H: tl.constexpr,
        W: tl.constexpr,
        N: tl.constexpr,
        S0: tl.constexpr,
        S1: tl.constexpr,
        S2: tl.constexpr,
        S3: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < 2 * N * 128 * H * W
        d = i % 32
        p = i // 32 % (H * W)
        head = i // (32 * H * W) % 4
        b = i // (128 * H * W) % N
        kv = i // (N * 128 * H * W)
        fx = tl.load(Flow + (b * 2) * H * W + p, ok, 0)
        fy = tl.load(Flow + (b * 2 + 1) * H * W + p, ok, 0)
        x = tl.minimum(tl.maximum(p % W + fx, 0.0), W - 1.0)
        y = tl.minimum(tl.maximum(p // W + fy, 0.0), H - 1.0)
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        x1 = tl.minimum(x0 + 1, W - 1)
        y1 = tl.minimum(y0 + 1, H - 1)
        wx = x - x0
        wy = y - y0
        base = b * S0 + (kv * 128 + head * 32 + d) * S1
        a = tl.load(X + base + y0 * S2 + x0 * S3, ok, 0).to(tl.float32)
        c = tl.load(X + base + y1 * S2 + x0 * S3, ok, 0).to(tl.float32)
        e = tl.load(X + base + y1 * S2 + x1 * S3, ok, 0).to(tl.float32)
        z = tl.load(X + base + y0 * S2 + x1 * S3, ok, 0).to(tl.float32)
        value = (
            a * ((1 - wx) * (1 - wy))
            + z * (wx * (1 - wy))
            + c * ((1 - wx) * wy)
            + e * (wx * wy)
        )
        tl.store(Y + i, value, ok)

    @triton.jit
    def _kv_warp_backward_kernel(
        G: tl.tensor,
        Flow: tl.tensor,
        DX: tl.tensor,
        H: tl.constexpr,
        W: tl.constexpr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < 2 * N * 128 * H * W
        d = i % 32
        p = i // 32 % (H * W)
        head = i // (32 * H * W) % 4
        b = i // (128 * H * W) % N
        kv = i // (N * 128 * H * W)
        fx = tl.load(Flow + (b * 2) * H * W + p, ok, 0)
        fy = tl.load(Flow + (b * 2 + 1) * H * W + p, ok, 0)
        x = tl.minimum(tl.maximum(p % W + fx, 0.0), W - 1.0)
        y = tl.minimum(tl.maximum(p // W + fy, 0.0), H - 1.0)
        x0 = tl.floor(x).to(tl.int32)
        y0 = tl.floor(y).to(tl.int32)
        x1 = tl.minimum(x0 + 1, W - 1)
        y1 = tl.minimum(y0 + 1, H - 1)
        wx = x - x0
        wy = y - y0
        g = tl.load(G + i, ok, 0).to(tl.float32)
        base = b * H * W * 256 + kv * 128 + head * 32 + d
        a = (1 - wx) * (1 - wy)
        z = wx * (1 - wy)
        c = (1 - wx) * wy
        e = wx * wy
        tl.atomic_add(DX + base + (y0 * W + x0) * 256, g * a, ok & (a != 0))
        tl.atomic_add(DX + base + (y0 * W + x1) * 256, g * z, ok & (z != 0))
        tl.atomic_add(DX + base + (y1 * W + x0) * 256, g * c, ok & (c != 0))
        tl.atomic_add(DX + base + (y1 * W + x1) * 256, g * e, ok & (e != 0))

    @triton_op("rdual_standalone::flow_kv", mutates_args={})
    def kv_warp(
        x: torch.Tensor, flow: torch.Tensor, height: int, width: int, block: int = 256
    ) -> torch.Tensor:
        n = x.shape[0]
        out = torch.empty((2, n, 4, height * width, 32), device=x.device, dtype=x.dtype)
        wrap_triton(_kv_warp_forward_kernel)[(triton.cdiv(out.numel(), block),)](
            x,
            flow,
            out,
            height,
            width,
            n,
            *x.stride(),
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out

    @triton_op("rdual_standalone::flow_kv_backward", mutates_args={})
    def kv_warp_backward(
        g: torch.Tensor, flow: torch.Tensor, height: int, width: int, block: int = 256
    ) -> torch.Tensor:
        n = g.shape[1]
        dx = torch.zeros((n, height, width, 256), device=g.device, dtype=torch.float32)
        wrap_triton(_kv_warp_backward_kernel)[(triton.cdiv(g.numel(), block),)](
            g.contiguous(),
            flow,
            dx,
            height,
            width,
            n,
            block,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return dx.permute(0, 3, 1, 2).to(g.dtype)

    def _kv_warp_setup_context(
        ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        ctx.save_for_backward(inputs[1])
        ctx.geometry = inputs[2:]

    def _kv_warp_backward(ctx: Any, g: torch.Tensor) -> tuple[Any, ...]:
        return (
            kv_warp_backward(g, ctx.saved_tensors[0], *ctx.geometry),
            None,
            None,
            None,
            None,
        )

    torch.library.register_autograd(
        kv_warp, _kv_warp_backward, setup_context=_kv_warp_setup_context
    )


# Temporal stage: PyTorch reference, checkpoint policy and state fusion.


def _temporal_checkpoint_policy(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
    # The spatial policy already retains SDPA/Flex outputs. Retain the temporal
    # aggregate and LSE too; recompute projections rather than the donor scan.
    if str(op) == "rdual_standalone.attention.default":
        return CheckpointPolicy.MUST_SAVE
    return _selective_checkpoint_policy(ctx, op, *args, **kwargs)


def temporal_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    table: Tensor,
    null_key: Tensor,
    null_bias: Tensor,
    slot_bias: Tensor,
    valid: Tensor,
    height: int,
    width: int,
    radius: int,
) -> Tensor:
    """Chunked exact query-centered attention, one joint zero-value null.

    Q is B,heads,HW,D; K/V are B,slots,heads,HW,D. Union tiles are only
    execution units: each query retains its independently anchored support.
    """
    batch, heads, _, dim = q.shape
    slots = k.shape[1]
    tile = 8
    ny, nx = math.ceil(height / tile), math.ceil(width / tile)
    uy, ux = min(height, tile + 2 * radius), min(width, tile + 2 * radius)
    wy, wx = min(height, 2 * radius + 1), min(width, 2 * radius + 1)
    axis = torch.arange(tile, device=q.device)
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    row, col = row.flatten(), col.flatten()
    kr, kc = torch.meshgrid(
        torch.arange(uy, device=q.device),
        torch.arange(ux, device=q.device),
        indexing="ij",
    )
    kr, kc = kr.flatten(), kc.flatten()
    outputs = []
    precision = torch.float64 if q.dtype == torch.float64 else torch.float32
    for first in range(0, ny * nx, 4):
        tiles = torch.arange(first, min(first + 4, ny * nx), device=q.device)
        count = tiles.shape[0]
        py = (tiles // nx)[:, None] * tile + row[None]
        px = (tiles % nx)[:, None] * tile + col[None]
        qi = py.clamp_max(height - 1) * width + px.clamp_max(width - 1)
        ky = ((tiles // nx) * tile - radius).clamp(0, height - uy)[:, None] + kr[None]
        kx = ((tiles % nx) * tile - radius).clamp(0, width - ux)[:, None] + kc[None]
        ki = ky * width + kx
        sy = (py - radius).clamp(0, height - wy)
        sx = (px - radius).clamp(0, width - wx)
        mask = (
            (ky[:, None] >= sy[:, :, None])
            & (ky[:, None] < sy[:, :, None] + wy)
            & (kx[:, None] >= sx[:, :, None])
            & (kx[:, None] < sx[:, :, None] + wx)
        )
        dy, dx = py[:, :, None] - ky[:, None], px[:, :, None] - kx[:, None]
        ti = ((dy + 2 * radius) * (4 * radius + 1) + dx + 2 * radius).clamp(
            0, (4 * radius + 1) ** 2 - 1
        )
        bias = table[:, ti].permute(1, 0, 2, 3).to(precision)
        bias = bias[None, :, :, :, None, :] + slot_bias.T[None, None, :, None, :, None]
        mask = mask[None, :, None, :, None, :] & valid[:, None, None, None, :, None]
        bias = bias.expand(batch, -1, -1, -1, -1, -1).masked_fill(~mask, -torch.inf)
        bias = bias.reshape(batch * count, heads, tile * tile, slots * uy * ux)
        bias = torch.cat(
            (
                bias,
                null_bias[None, :, None, None].expand(
                    batch * count, heads, tile * tile, 1
                ),
            ),
            -1,
        )
        qt = (
            q[:, :, qi, :]
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * count, heads, tile * tile, dim)
        )
        kt = (
            k[:, :, :, ki, :]
            .permute(0, 3, 2, 1, 4, 5)
            .reshape(batch * count, heads, slots * uy * ux, dim)
        )
        vt = (
            v[:, :, :, ki, :]
            .permute(0, 3, 2, 1, 4, 5)
            .reshape(batch * count, heads, slots * uy * ux, dim)
        )
        nk = null_key.to(k.dtype)[None, :, None, :].expand(batch * count, -1, -1, -1)
        kt = torch.cat((kt, nk), -2)
        vt = torch.cat((vt, torch.zeros_like(nk)), -2)
        # Keep the learned bias and normalization in FP32 even when a CUDA SDPA
        # backend requires bias and operands to have matching storage dtypes.
        with torch.autocast(q.device.type, enabled=False):
            out = F.scaled_dot_product_attention(
                qt.to(precision), kt.to(precision), vt.to(precision), attn_mask=bias
            ).to(v.dtype)
        outputs.append(out.reshape(batch, count, heads, tile, tile, dim))
    out = torch.cat(outputs, 1).reshape(batch, ny, nx, heads, tile, tile, dim)
    out = out.permute(0, 3, 1, 4, 2, 5, 6).reshape(
        batch, heads, ny * tile, nx * tile, dim
    )
    return out[:, :, :height, :width].reshape(batch, heads, height * width, dim)


def _sample_points(flow: Tensor) -> Tensor:
    """Absolute x/y source positions, N,2,H,W, of N,2,H,W target-to-source flow.

    Pixel i of a row sits at coordinate i; the frame spans [0, size - 1].
    """
    height, width = flow.shape[-2:]
    y, u = torch.meshgrid(
        torch.arange(height, dtype=flow.dtype, device=flow.device),
        torch.arange(width, dtype=flow.dtype, device=flow.device),
        indexing="ij",
    )
    return torch.stack((u, y), 0)[None] + flow


def warp_image(image: Tensor, flow: Tensor) -> Tensor:
    """Bilinear border warp of N,C,H,W by N,2,H,W flow in pixels, same dtype."""
    height, width = image.shape[-2:]
    xy = _sample_points(flow)
    grid = torch.stack(
        ((2 * xy[:, 0] + 1) / width - 1, (2 * xy[:, 1] + 1) / height - 1), -1
    )
    return torch.nn.functional.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def warp_reference(x: Tensor, flow: Tensor, height: int, width: int) -> Tensor:
    """Packed N,4,HW,32 features; fixed guidance, differentiable feature values."""
    precision = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(x.device.type, enabled=False):
        image = x.transpose(-1, -2).reshape(x.shape[0], 128, height, width)
        sampled = warp_image(image.to(precision), flow.to(precision))
        return (
            sampled.reshape(x.shape[0], 4, 32, height * width)
            .transpose(-1, -2)
            .contiguous()
            .to(x.dtype)
        )


def alignment_cues(
    target: Tensor, source: Tensor, flow: Tensor, reverse: Tensor
) -> Tensor:
    """The three alignment cues of a donor on the target grid, N,3,H,W.

    target and source are N,C,H,W images; flow is N,2,H,W target-to-source
    flow in pixels and reverse the source-to-target flow on the source grid.
    The cues are the channel-mean absolute difference between the warped
    source and the target, the forward-backward inconsistency
    |flow(p) + reverse(p + flow(p))| in pixels and 1 where p + flow(p) leaves
    the frame, else 0.
    """
    height, width = flow.shape[-2:]
    photometric = (warp_image(source, flow) - target).abs().mean(1, keepdim=True)
    inconsistency = (flow + warp_image(reverse, flow)).norm(dim=1, keepdim=True)
    xy = _sample_points(flow)
    outside = (
        (xy[:, 0] < 0)
        | (xy[:, 0] > width - 1)
        | (xy[:, 1] < 0)
        | (xy[:, 1] > height - 1)
    )
    return torch.cat((photometric, inconsistency, outside[:, None].to(flow.dtype)), 1)


_INDICES = ((0, 1, 2, 3), (3, 4, 5), (0, 1, 2), (0, 1, 2, 3, 4, 5))
_PAST, _FUTURE = 0, 1

TemporalBackend = Literal["auto", "reference", "triton"]


class StateFusion(nn.Module):
    """Query-centered temporal attention with carried trunk states as donors.

    Slot 0 is the past state and slot 1 the future state; a call may present
    either or both. Donors use their own norm: they leave the trunk's end,
    while the query is taken after the first group.

    The last channel of every head is a bias lane. The query's lane holds
    sqrt(32), which the 1/sqrt(32) score scale of both attention paths turns
    into a weight of one, so the key's lane adds to the logit as it stands:
    the key projection's own lane (a linear source quality of the donor,
    warped with the key) plus cue_score of the donor's alignment cues. Both
    start at zero.
    """

    def __init__(self, backend: TemporalBackend = "auto") -> None:
        super().__init__()
        if backend not in ("auto", "reference", "triton"):
            raise ValueError("temporal_backend must be auto, reference or triton")
        if backend == "triton" and not _HAS_DIRECT_ATTENTION:
            raise RuntimeError(
                "RDUAL Triton temporal kernels are unavailable: "
                f"{_DIRECT_ATTENTION_ERROR}. Use temporal_backend='auto' or "
                "'reference', or provide the required Triton/PyTorch APIs."
            )
        slots, radius = 2, 3
        self.radius, self.backend = radius, backend
        self.norm = TokenNorm(128, 1e-5)
        self.query = nn.Conv2d(128, 128, 1, bias=False)
        self.key = nn.Conv2d(128, 128, 1, bias=False)
        self.value = nn.Conv2d(128, 128, 1, bias=False)
        self.output = nn.Conv2d(128, 128, 1, bias=False)
        for layer in (self.query, self.key, self.value):
            nn.init.trunc_normal_(layer.weight, std=0.02)
        with torch.no_grad():
            self.key.weight[31::32].zero_()
        self.cue_score = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 4, 1)
        )
        nn.init.zeros_(self.cue_score[2].weight)
        nn.init.zeros_(self.cue_score[2].bias)
        nn.init.zeros_(self.output.weight)
        angles = torch.arange(8) * math.pi * (3 - math.sqrt(5))
        lengths = torch.logspace(
            math.log10(2 * math.pi / 128), math.log10(math.pi / 2), 8
        )
        self.frequency = nn.Parameter(
            lengths[:, None] * torch.stack((angles.cos(), angles.sin()), -1)
        )
        self.cosine = nn.Parameter(torch.zeros(4, 8))
        self.sine = nn.Parameter(torch.zeros(4, 8))
        self.null_key = nn.Parameter(torch.zeros(4, 32))
        self.slot_bias = nn.Parameter(torch.zeros(slots, 4))
        kappa = torch.tensor((0.25, 1.0, 4.0, 16.0))
        self.register_buffer("kappa", kappa, persistent=False)
        axis = torch.arange(-radius, radius + 1)
        mass = (
            torch.exp(-kappa[:, None] * axis.square()[None] / radius**2)
            .sum(-1)
            .square()
        )
        self.null_bias = nn.Parameter(torch.log(slots * mass / 4))
        self.state_norm = TokenNorm(128, 1e-5)

    @property
    def kernels_available(self) -> bool:
        """Whether this back end runs the Triton kernels in this process."""
        return self.backend == "triton" or (
            self.backend == "auto" and _HAS_DIRECT_ATTENTION
        )

    def use_kernels(self, x: Tensor, flow: Tensor) -> bool:
        """The per-call back-end choice for features x and guidance flow.

        "auto" runs the kernels for CUDA features that are not FP64, with FP32
        guidance, outside deterministic mode, which is read per call; otherwise
        it runs the reference. "triton" takes them for every CUDA call and
        raises where they cannot run.
        """
        if not x.is_cuda or not self.kernels_available:
            return False
        return self.backend == "triton" or (
            not torch.are_deterministic_algorithms_enabled()
            and x.dtype != torch.float64
            and flow.dtype == torch.float32
        )

    def relative_table(self) -> Tensor:
        precision = (
            torch.float64 if self.frequency.dtype == torch.float64 else torch.float32
        )
        with torch.autocast(self.frequency.device.type, enabled=False):
            axis = torch.arange(
                -2 * self.radius,
                2 * self.radius + 1,
                device=self.frequency.device,
                dtype=precision,
            )
            dy, dx = torch.meshgrid(axis, axis, indexing="ij")
            offsets = torch.stack((dy.flatten(), dx.flatten()), -1)
            phase = offsets @ self.frequency.to(precision).T
            return (
                self.cosine.to(precision) @ phase.cos().T
                + self.sine.to(precision) @ phase.sin().T
                - self.kappa.to(precision)[:, None]
                * offsets.square().sum(-1)[None]
                / self.radius**2
            )

    def align_projections(
        self, k: Tensor, v: Tensor, flow: Tensor, height: int, width: int
    ) -> tuple[Tensor, Tensor]:
        batch, slots = k.shape[:2]
        guidance = flow.reshape(batch * slots, 2, height, width).contiguous()
        return tuple(
            warp_reference(
                t.reshape(batch * slots, 4, height * width, 32), guidance, height, width
            ).reshape(t.shape)
            for t in (k, v)
        )

    def forward(
        self,
        center: Tensor,
        states: Tensor,
        valid: Tensor,
        flow: Tensor,
        slots: Tensor,
    ) -> Tensor:
        """Fuse the donors into center, B,128,Htrunk,Wtrunk.

        flow is B,donors,5,Htrunk,Wtrunk: detached FP32 x/y warp coordinates,
        then the donor's three alignment cues.
        """
        batch, count, _, height, width = states.shape
        if slots.shape != (count,) or flow.shape != (batch, count, 5, height, width):
            raise ValueError("Flow shape must match B,donors,5,Htrunk,Wtrunk")
        coordinates, cues = flow[:, :, :2], flow[:, :, 2:]
        query = (
            self.query(self.norm(center))
            .reshape(batch, 4, 32, height * width)
            .transpose(-1, -2)
        )
        lane = query.new_full((batch, 4, height * width, 1), math.sqrt(32))
        query = torch.cat((query[..., :31], lane), -1)
        normalized = self.state_norm(states.reshape(batch * count, 128, height, width))
        kernels = self.use_kernels(center, flow)
        if kernels:
            if torch.are_deterministic_algorithms_enabled():
                raise RuntimeError(
                    "Flow-warp atomic adjoints require nondeterministic mode"
                )
            if flow.dtype != torch.float32:
                raise ValueError("Flow warp requires FP32 coordinates")
            projected = F.conv2d(
                normalized, torch.cat((self.key.weight, self.value.weight), dim=0)
            )
            if projected.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("CUDA warp supports FP16/BF16/FP32 features")
            packed = kv_warp(
                projected,
                coordinates.reshape(batch * count, 2, height, width).contiguous(),
                height,
                width,
            )
            key, value = (
                t.reshape(batch, count, 4, height * width, 32) for t in packed.unbind(0)
            )
        else:
            key, value = self.align_projections(
                *(
                    layer(normalized)
                    .reshape(batch, count, 4, 32, height * width)
                    .transpose(-1, -2)
                    .contiguous()
                    for layer in (self.key, self.value)
                ),
                coordinates,
                height,
                width,
            )
        # The cue score joins the key's lane on the query grid, after the warp.
        score = self.cue_score(
            cues.reshape(batch * count, 3, height, width).to(
                self.cue_score[0].weight.dtype
            )
        ).reshape(batch, count, 4, height * width, 1)
        key = torch.cat((key[..., :31], key[..., 31:] + score.to(key.dtype)), -1)
        attention = temporal_attention if kernels else temporal_reference
        out = attention(
            query,
            key,
            value,
            self.relative_table(),
            self.null_key,
            self.null_bias,
            self.slot_bias[slots],
            valid,
            height,
            width,
            self.radius,
        )
        return center + self.output(
            out.transpose(-1, -2).reshape(batch, 128, height, width)
        )


Encoded = tuple[Tensor, ...]
# Position of the trunk features in the tuple encode() returns.
_ENCODED_TRUNK = 3


class StepFunctions(NamedTuple):
    """The per-frame callables of a clip, eager or compiled.

    Steps take the future donor as three trailing positional arguments
    (state, flow, valid). compose is the fold_cache() composition, None for
    compose_kernels().
    """

    encode: Callable[[Tensor], Encoded]
    full: Callable[..., tuple[Tensor | None, Tensor]]
    reverse: Callable[..., Tensor]
    forward_state: Callable[..., Tensor]
    compose: Callable[[], list[Tensor]] | None


def _plain(encoded: Encoded) -> Encoded:
    return tuple(x.detach() for x in encoded)


def _marks_cuda_graph_steps(frames: Tensor) -> bool:
    """Whether backward_clip() marks CUDA-graph steps: CUDA frames only."""
    return frames.is_cuda


# Frozen NeuFlow v2 optical flow (Apache-2.0, see the module docstring) for
# the clip guidance.

# Adjacent frame pairs per flow call (each runs in both directions)
_FLOW_PAIRS = 16


def _flow_pairs_per_call(padded_h: int, padded_w: int) -> int:
    """Pairs per flow call for a padded frame size, at least one.

    The 1/8-scale all-pairs correlation holds (h/8 * w/8)^2 float32 values
    per flow and a pair is two flows; a call's correlation stays within
    2 GiB where one pair allows it.
    """
    cells = (padded_h // 8) * (padded_w // 8)
    return max(1, min(_FLOW_PAIRS, 2**31 // (2 * 4 * cells * cells)))


_NF_FEATURE_DIM_S16 = 128
_NF_CONTEXT_DIM_S16 = 64
_NF_ITER_CONTEXT_DIM_S16 = 64
_NF_FEATURE_DIM_S8 = 128
_NF_CONTEXT_DIM_S8 = 64
_NF_ITER_CONTEXT_DIM_S8 = 64
_NF_FEATURE_DIM_S1 = 128
_NF_CORR_RADIUS = 4
_NF_ITERS_S16 = 1
_NF_ITERS_S8 = 8


def _neuflow_coords_grid(
    batch: int, height: int, width: int, device: torch.device
) -> Tensor:
    """Pixel coordinates B,2,H,W with x first."""
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0)[None].repeat(batch, 1, 1, 1)


class _NeuFlowConvBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, bias=False
        )
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.norm1 = nn.BatchNorm2d(out_planes)
        self.norm2 = nn.BatchNorm2d(out_planes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        return self.relu(self.norm2(self.conv2(x)))


class _NeuFlowEncoder(nn.Module):
    """Features at 1/8 and 1/16 scale; the 1/16 ones end with a position grid."""

    def __init__(self) -> None:
        super().__init__()
        self.block_8_1 = _NeuFlowConvBlock(3, _NF_FEATURE_DIM_S8 * 2, 8, 4, 2)
        self.block_8_2 = _NeuFlowConvBlock(3, _NF_FEATURE_DIM_S8, 6, 2, 2)
        self.block_cat_8 = _NeuFlowConvBlock(
            _NF_FEATURE_DIM_S8 * 3, _NF_FEATURE_DIM_S8 + _NF_CONTEXT_DIM_S8, 3, 1, 1
        )
        self.block_16_1 = _NeuFlowConvBlock(3, _NF_FEATURE_DIM_S16, 6, 2, 2)
        self.block_8_16 = _NeuFlowConvBlock(
            _NF_FEATURE_DIM_S8 + _NF_CONTEXT_DIM_S8, _NF_FEATURE_DIM_S16, 6, 2, 2
        )
        self.block_cat_16 = _NeuFlowConvBlock(
            _NF_FEATURE_DIM_S16 * 2,
            _NF_FEATURE_DIM_S16 + _NF_CONTEXT_DIM_S16 - 2,
            3,
            1,
            1,
        )

    def forward(self, img: Tensor, pos_s16: Tensor) -> tuple[Tensor, Tensor]:
        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x_8 = self.block_8_1(img)
        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x_8_2 = self.block_8_2(img)
        x_8 = self.block_cat_8(torch.cat([x_8, x_8_2], dim=1))
        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        x_16 = self.block_16_1(img)
        x_16_2 = self.block_8_16(x_8)
        x_16 = self.block_cat_16(torch.cat([x_16, x_16_2], dim=1))
        return torch.cat([x_16, pos_s16], dim=1), x_8


class _NeuFlowTransformerLayer(nn.Module):
    """Single-head cross attention with a post-normalized FFN."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.merge = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        message = F.scaled_dot_product_attention(
            self.q_proj(source), self.k_proj(target), self.v_proj(target)
        )
        message = self.norm1(self.merge(message))
        message = self.norm2(self.mlp(torch.cat([source, message], dim=-1)))
        return source + message


class _NeuFlowFeatureAttention(nn.Module):
    """Each image of the pair batch [first images, second images] attends to its partner."""

    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_NeuFlowTransformerLayer(dim) for _ in range(num_layers)]
        )
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, features: Tensor) -> Tensor:
        b, c, h, w = features.shape
        tokens = features.flatten(-2).permute(0, 2, 1)
        partners = torch.cat(tokens.chunk(chunks=2, dim=0)[::-1], dim=0)
        for layer in self.layers:
            tokens = layer(tokens, partners)
            partners = torch.cat(tokens.chunk(chunks=2, dim=0)[::-1], dim=0)
        tokens = tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return self.norm(tokens)


class _NeuFlowRefineBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x))


class _NeuFlowRefine(nn.Module):
    """One refinement step: new iteration context and a flow update."""

    def __init__(
        self, context_dim: int, iter_context_dim: int, num_layers: int, inter_dim: int
    ) -> None:
        super().__init__()
        corr_dim = (_NF_CORR_RADIUS * 2 + 1) ** 2
        self.conv1 = _NeuFlowRefineBlock(
            corr_dim + context_dim + iter_context_dim + 2 + 1,
            context_dim + iter_context_dim,
        )
        self.conv2 = _NeuFlowRefineBlock(context_dim + iter_context_dim, inter_dim)
        self.conv_layers = nn.ModuleList(
            [_NeuFlowRefineBlock(inter_dim, inter_dim) for _ in range(num_layers)]
        )
        self.conv3 = nn.Conv2d(inter_dim, iter_context_dim + 2, 3, 1, 1)
        self.hidden_act = nn.Hardtanh(min_val=-4.0, max_val=4.0)

    def forward(
        self,
        corrs: Tensor,
        context: Tensor,
        iter_context: Tensor,
        flow: Tensor,
        radius_emb: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x = torch.cat([corrs, context, iter_context, flow, radius_emb], dim=1)
        x = self.conv2(self.conv1(x))
        for layer in self.conv_layers:
            x = layer(x)
        x = self.conv3(x)
        return self.hidden_act(x[:, 2:]), x[:, :2]


class _NeuFlowUpSample(nn.Module):
    """Convex 8x upsampling of a flow field guided by full-size image features."""

    def __init__(self, feature_dim: int, upsample_factor: int) -> None:
        super().__init__()
        self.upsample_factor = upsample_factor
        self.conv1 = nn.Conv2d(2 + feature_dim, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, upsample_factor**2 * 9, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature: Tensor, flow: Tensor) -> Tensor:
        k = self.upsample_factor
        concat = torch.cat((flow, feature), dim=1)
        mask = self.conv3(self.relu(self.conv2(self.relu(self.conv1(concat)))))
        b, _, h, w = flow.shape
        mask = torch.softmax(mask.view(b, 1, 9, k, k, h, w), dim=2)
        up_flow = F.unfold(flow, [3, 3], padding=1).view(b, 2, 9, 1, 1, h, w)
        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(b, 2, k * h, k * w)


def _neuflow_correlation(feature0: Tensor, feature1: Tensor) -> Tensor:
    """All-pairs correlation, one B*H*W,1,H,W map per pixel of feature0."""
    b, c, h, w = feature0.shape
    corr = torch.matmul(
        feature0.view(b, c, h * w).transpose(1, 2), feature1.view(b, c, h * w)
    )
    # In place: the volume is the largest tensor of the flow.
    return corr.view(b * h * w, 1, h, w).div_(math.sqrt(c))


def _neuflow_lookup(corr: Tensor, grid: Tensor, flow: Tensor) -> Tensor:
    """The (2r+1)² correlation window around each pixel's flow target."""
    b, _, h, w = flow.shape
    radius = torch.linspace(
        -_NF_CORR_RADIUS, _NF_CORR_RADIUS, 2 * _NF_CORR_RADIUS + 1, device=flow.device
    )
    delta = torch.stack(torch.meshgrid(radius, radius, indexing="ij"), dim=-1)
    coords = (grid + flow).permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
    x, y = (coords + delta[None]).split((1, 1), dim=-1)
    sample = torch.cat(
        (2 * x / (corr.shape[-1] - 1) - 1, 2 * y / (corr.shape[-2] - 1) - 1), -1
    )
    # bilinear, zero padding, align_corners: the sampler the original selects
    # with cuDNN disabled
    sampled = torch.grid_sampler_2d(corr, sample, 0, 0, True)
    return sampled.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class NeuFlowV2(nn.Module):
    """NeuFlow v2 inference: flow from img0 to img1 at full size, N,2,H,W.

    Images are N,3,H,W RGB in 0..255, as in the original, with H and W
    multiples of 16.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _NeuFlowEncoder()
        self.cross_attn_s16 = _NeuFlowFeatureAttention(
            _NF_FEATURE_DIM_S16 + _NF_CONTEXT_DIM_S16, num_layers=2
        )
        self.merge_s8 = nn.Sequential(
            nn.Conv2d(
                _NF_FEATURE_DIM_S16 + _NF_FEATURE_DIM_S8,
                _NF_FEATURE_DIM_S8,
                3,
                1,
                1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(_NF_FEATURE_DIM_S8, _NF_FEATURE_DIM_S8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(_NF_FEATURE_DIM_S8),
        )
        self.context_merge_s8 = nn.Sequential(
            nn.Conv2d(
                _NF_CONTEXT_DIM_S16 + _NF_CONTEXT_DIM_S8,
                _NF_CONTEXT_DIM_S8,
                3,
                1,
                1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(_NF_CONTEXT_DIM_S8, _NF_CONTEXT_DIM_S8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(_NF_CONTEXT_DIM_S8),
        )
        self.refine_s16 = _NeuFlowRefine(
            _NF_CONTEXT_DIM_S16, _NF_ITER_CONTEXT_DIM_S16, num_layers=5, inter_dim=128
        )
        self.refine_s8 = _NeuFlowRefine(
            _NF_CONTEXT_DIM_S8, _NF_ITER_CONTEXT_DIM_S8, num_layers=5, inter_dim=96
        )
        self.conv_s8 = _NeuFlowConvBlock(3, _NF_FEATURE_DIM_S1, 8, 8, 0)
        self.upsample_s8 = _NeuFlowUpSample(_NF_FEATURE_DIM_S1, upsample_factor=8)
        self._grids_key: tuple[int, int, int, torch.device] | None = None
        self._grids: dict[str, Tensor] = {}

    def _prepare_grids(
        self, batch: int, height: int, width: int, device: torch.device
    ) -> dict[str, Tensor]:
        key = (batch, height, width, device)
        if key != self._grids_key:
            h16, w16, h8, w8 = height // 16, width // 16, height // 8, width // 8
            ys, xs = torch.meshgrid(
                torch.arange(h16, dtype=torch.float32, device=device),
                torch.arange(w16, dtype=torch.float32, device=device),
                indexing="ij",
            )
            pos = torch.stack([ys - h16 / 2, xs - w16 / 2])
            grid_s16 = _neuflow_coords_grid(batch, h16, w16, device)
            self._grids = {
                "pos_s16": pos[None].repeat(batch * 2, 1, 1, 1),
                "grid_s16": grid_s16,
                "flat_grid_s16": grid_s16.view(batch, 2, -1).permute(0, 2, 1),
                "grid_s8": _neuflow_coords_grid(batch, h8, w8, device),
                "radius_s16": torch.full(
                    (batch, 1, h16, w16), float(_NF_CORR_RADIUS), device=device
                ),
                "radius_s8": torch.full(
                    (batch, 1, h8, w8), float(_NF_CORR_RADIUS), device=device
                ),
            }
            self._grids_key = key
        return self._grids

    def forward(self, img0: Tensor, img1: Tensor) -> Tensor:
        b, _, height, width = img0.shape
        grids = self._prepare_grids(b, height, width, img0.device)
        img0 = img0 / 255.0
        img1 = img1 / 255.0

        features_s16, features_s8 = self.backbone(
            torch.cat([img0, img1], dim=0), grids["pos_s16"]
        )
        features_s16 = self.cross_attn_s16(features_s16)
        context_s16, features_s16 = torch.split(
            features_s16, [_NF_CONTEXT_DIM_S16, _NF_FEATURE_DIM_S16], dim=1
        )
        context_s16 = torch.relu(context_s16.chunk(chunks=2, dim=0)[0])
        context_s8, features_s8 = torch.split(
            features_s8, [_NF_CONTEXT_DIM_S8, _NF_FEATURE_DIM_S8], dim=1
        )
        context_s8 = torch.relu(context_s8.chunk(chunks=2, dim=0)[0])

        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)
        correspondence = F.scaled_dot_product_attention(
            feature0_s16.flatten(-2).permute(0, 2, 1),
            feature1_s16.flatten(-2).permute(0, 2, 1),
            grids["flat_grid_s16"],
        )
        h16, w16 = feature0_s16.shape[-2:]
        flow = (
            correspondence.view(b, h16, w16, 2).permute(0, 3, 1, 2) - grids["grid_s16"]
        )

        corr_s16 = _neuflow_correlation(feature0_s16, feature1_s16)
        iter_context = torch.zeros(
            b, _NF_ITER_CONTEXT_DIM_S16, h16, w16, device=img0.device
        )
        for _ in range(_NF_ITERS_S16):
            corrs = _neuflow_lookup(corr_s16, grids["grid_s16"], flow)
            iter_context, delta_flow = self.refine_s16(
                corrs, context_s16, iter_context, flow, grids["radius_s16"]
            )
            flow = flow + delta_flow

        flow = F.interpolate(flow, scale_factor=2, mode="nearest") * 2
        features_s16 = F.interpolate(features_s16, scale_factor=2, mode="nearest")
        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))
        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)
        corr_s8 = _neuflow_correlation(feature0_s8, feature1_s8)
        context_s16 = F.interpolate(context_s16, scale_factor=2, mode="nearest")
        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))

        iter_context = torch.zeros(
            b, _NF_ITER_CONTEXT_DIM_S8, *feature0_s8.shape[-2:], device=img0.device
        )
        for _ in range(_NF_ITERS_S8):
            corrs = _neuflow_lookup(corr_s8, grids["grid_s8"], flow)
            iter_context, delta_flow = self.refine_s8(
                corrs, context_s8, iter_context, flow, grids["radius_s8"]
            )
            flow = flow + delta_flow

        return self.upsample_s8(self.conv_s8(img0), flow) * 8


@store_hyperparameters()
class RDUAL(nn.Module):
    """Recurrent DUAL Light; see the module docstring.

    Removed constructor options are not accepted. prepare_for_export() is
    irreversible and must operate on an eval copy, never the training instance.
    """

    def __init__(
        self,
        *,
        scale: int = 4,
        use_checkpoint: bool = True,
        dynamic_experts: int = 4,
        c_windows: tuple[int, int] = (32, 64),
        temporal_backend: TemporalBackend = "auto",
        attn_type: Literal["auto", "masked", "hybrid"] = "auto",
    ) -> None:
        super().__init__()
        if type(use_checkpoint) is not bool:
            raise ValueError("use_checkpoint must be a bool")
        if type(dynamic_experts) is not int or dynamic_experts < 2:
            raise ValueError("dynamic_experts must be an integer >= 2 (not bool)")
        if scale not in (1, 2, 4):
            raise ValueError("RDUAL requires scale 1, 2 or 4")
        if len(c_windows) != 2 or min(c_windows) < 1:
            raise ValueError("two positive context windows required")
        if attn_type not in ("auto", "masked", "hybrid"):
            raise ValueError("attn_type must be auto/masked/hybrid")
        if attn_type == "hybrid" and flex_attention is None:
            raise ImportError("RDUAL hybrid attention requires PyTorch Flex Attention")
        # Only an explicit "hybrid" turns a missing hybrid requirement into an error.
        self.require_hybrid = attn_type == "hybrid"
        if attn_type == "auto":
            attn_type = "hybrid" if flex_attention is not None else "masked"
        self.attn_type = attn_type
        self.scale, self.internal_scale = scale, 2 * scale
        self.use_checkpoint = use_checkpoint
        self.register_buffer(
            "mean", torch.tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        )
        # Construction follows DUAL Light's so a seed gives its initial weights:
        # full groups, a three-channel stem and the plain tail are built and
        # initialized first, then replaced by the pruned and packed modules.
        channels = 128
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.groups = nn.Sequential(
            *(_Group(channels, c_windows, use_checkpoint) for _ in _INDICES)
        )
        self.final_norm = TokenNorm(channels, 1e-5)
        self.trunk_end = nn.Conv2d(channels, channels, 3, padding=1)
        tail: list[nn.Module] = [
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
        ]
        for stage in (2, 2) if scale == 4 else ((2,) if scale == 2 else ()):
            tail.extend(
                (
                    nn.Conv2d(64, stage**2 * 64, 3, padding=1),
                    nn.PixelShuffle(stage),
                )
            )
        tail.append(nn.Conv2d(64, 3, 3, padding=1))
        self.tail = nn.Sequential(*tail)
        self.apply(_init_weights)
        self.xr = CrossScaleRetrieval(channels)
        self.xg = SecondOctaveTransfer(channels) if self.internal_scale >= 4 else None
        for group in self.groups:
            for index in (2, 5):
                group.layers[index] = DualContextLayer(group.layers[index])
                group.layers[index].attention.attn_type = attn_type
                group.layers[index].attention.require_hybrid = self.require_hybrid
        for group, indices in zip(self.groups, _INDICES, strict=True):
            group.layers = nn.ModuleList(group.layers[i] for i in indices)
        # PixelUnshuffle2 input packing and the matching final 2x upsampling.
        self.stem = nn.Conv2d(12, channels, 3, padding=1)
        self.tail = nn.Sequential(
            *list(self.tail.children())[:-1],
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            self.tail[-1],
        )
        if self.xg is not None and self.internal_scale == 8:
            self.xg.out = nn.Conv2d(channels // 16, 12, 3, padding=1, bias=False)
            nn.init.zeros_(self.xg.out.weight)
        targets = [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, nn.Conv2d)
            and module.groups == 1
            and module.kernel_size == (3, 3)
        ]
        self.reparameterized_names = tuple(name for name, _ in targets)
        for name, module in targets:
            parent, _, leaf = name.rpartition(".")
            setattr(self.get_submodule(parent), leaf, EDBBReparamConv3x3(module))
        self.deployed = False
        with torch.random.fork_rng(devices=[]):
            for group in self.groups:
                group.conv = PremixedDynamicGroupConv3x3(group.conv, dynamic_experts)
        self.temporal = StateFusion(temporal_backend)
        # The tag names the back end that runs, not the requested one, so a
        # cache built without the kernels is never reused with them.
        backend = "triton" if self.temporal.kernels_available else "reference"
        self.compile_cache_key_tag = (
            f"dual-family-attention-v5:selective-checkpoint={use_checkpoint}"
            f":attn-type={attn_type}-rdual-{backend}-v5"
        )
        # The frozen flow estimator comes last, on a forked RNG, so a seed still
        # gives every other initial weight; load_flow_weights() fills it.
        with torch.random.fork_rng(devices=[]):
            self.flow = NeuFlowV2()
        self.flow.requires_grad_(False)
        self.flow.eval()
        self.register_buffer("flow_loaded", torch.tensor(False))

    def train(self, mode: bool = True) -> RDUAL:
        super().train(mode)
        # The flow estimator stays in eval mode: its BatchNorm statistics are fixed.
        self.flow.eval()
        return self

    def load_flow_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Load an upstream NeuFlow v2 state dict (no prefix) strictly into flow."""
        self.flow.load_state_dict(state_dict, strict=True)
        self.flow_loaded.fill_(True)

    @torch.no_grad()
    def guidance(self, frames: Tensor) -> Tensor:
        """The flow guidance of a B,T,3,H,W clip in 0..1: B,T-1,2,2,Htrunk,Wtrunk.

        Frozen NeuFlow v2 (one s16 and eight s8 iterations) on every adjacent
        pair in both directions, sampled at PU2 token centres (LR 2u+0.5,
        2v+0.5) and halved to trunk pixels. Entry [i,0] maps frame i+1 onto
        frame i, [i,1] frame i onto frame i+1. FP32 without gradients, with
        autocast and TF32 off inside and the caller's settings restored; any
        B, T and frame size (T == 1 gives an empty pair axis). Frames are
        padded to a multiple of 16 and at least 32 pixels a side, and run on
        the model's device in calls of at most _FLOW_PAIRS pairs, fewer for
        large frames (_flow_pairs_per_call); the result is on the frames'
        device.
        """
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("RDUAL input must be B,T,3,H,W")
        if not bool(self.flow_loaded):
            raise RuntimeError(
                "RDUAL has no NeuFlow v2 weights: call load_flow_weights() with the "
                "upstream state dict, or pass a guidance tensor instead"
            )
        batch, count, _, height, width = frames.shape
        h4, w4 = ((height + 3) // 4) * 4, ((width + 3) // 4) * 4
        if count < 2:
            return torch.zeros((batch, 0, 2, 2, h4 // 2, w4 // 2), device=frames.device)
        device = self.flow_loaded.device
        padded_h = max(32, (height + 15) // 16 * 16)
        padded_w = max(32, (width + 15) // 16 * 16)
        later = frames[:, 1:].flatten(0, 1)
        earlier = frames[:, :-1].flatten(0, 1)
        pairs = _flow_pairs_per_call(padded_h, padded_w)
        # The per-backend switches only: the global getters raise once a
        # process has set one backend on its own.
        matmul, conv = torch.backends.cuda.matmul, torch.backends.cudnn.conv
        matmul_precision, conv_precision = matmul.fp32_precision, conv.fp32_precision
        flows = []
        with torch.autocast(device.type, enabled=False):
            matmul.fp32_precision = "ieee"
            conv.fp32_precision = "ieee"
            try:
                for first in range(0, later.shape[0], pairs):
                    span = slice(first, first + pairs)
                    pair_later = later[span].to(device, torch.float32) * 255.0
                    pair_earlier = earlier[span].to(device, torch.float32) * 255.0
                    # direction 0 maps frame i+1 onto frame i, direction 1 the reverse
                    source = torch.stack((pair_later, pair_earlier), 1).flatten(0, 1)
                    target = torch.stack((pair_earlier, pair_later), 1).flatten(0, 1)
                    if padded_h != height or padded_w != width:
                        padding = (0, padded_w - width, 0, padded_h - height)
                        source = F.pad(source, padding, mode="replicate")
                        target = F.pad(target, padding, mode="replicate")
                    flow = self.flow(source, target)[:, :, :height, :width]
                    if h4 != height or w4 != width:
                        flow = F.pad(
                            flow, (0, w4 - width, 0, h4 - height), mode="replicate"
                        )
                    flow = F.interpolate(
                        flow,
                        size=(h4 // 2, w4 // 2),
                        mode="bilinear",
                        align_corners=False,
                    )
                    flows.append((flow * 0.5).to(frames.device))
            finally:
                matmul.fp32_precision = matmul_precision
                conv.fp32_precision = conv_precision
        return torch.cat(flows).reshape(batch, count - 1, 2, 2, h4 // 2, w4 // 2)

    def _folded(self) -> list[EDBBReparamConv3x3]:
        return [
            module
            for module in self.modules()
            if isinstance(module, EDBBReparamConv3x3)
            and module.direct.expand.weight.dtype == torch.float32
        ]

    def compose_kernels(self) -> list[Tensor]:
        """Every FP32 EDBB kernel, then its bias where present, in module order.

        fold_cache() calls this on entry and, for a tracking block, again on
        exit. Compiling it keeps the per-clip composition and its backward out
        of eager mode.
        """
        values: list[Tensor] = []
        with torch.autocast(device_type=self.mean.device.type, enabled=False):
            for module in self._folded():
                kernel, bias = module.compose(torch.float32)
                values.append(kernel)
                if bias is not None:
                    values.append(bias)
        return values

    @contextmanager
    def fold_cache(
        self, compose: Callable[[], list[Tensor]] | None = None
    ) -> Iterator[None]:
        """Compose every EDBB kernel once for the frames run inside the block.

        Entered with gradients enabled, frames convolve with leaf kernels whose
        gradients accumulate in place over the block's backward calls; leaving
        composes the kernels again with gradients and backpropagates the
        accumulated gradients through that composition at once, which by
        linearity gives the per-frame parameter gradients. The entry
        composition runs without gradients, so no graph spans the block. Every
        backward therefore belongs inside the block, parameters must not change
        inside it (step the optimizer afterwards), and a state carried into a
        later block must be detached. Entered under torch.no_grad() it composes
        once and only shares the kernels. Blocks do not nest, and an exception
        inside one skips the closing backward. compose replaces compose_kernels,
        normally with its compiled form; it is called once on entry and once
        more on leaving a tracking block.
        """
        modules = self._folded()
        if any(module.fold_active for module in modules):
            raise RuntimeError("fold_cache() is already active")
        track = torch.is_grad_enabled()
        composition = self.compose_kernels if compose is None else compose
        with torch.no_grad():
            values = iter(composition())
        leaves: list[Tensor] = []
        for module in modules:
            leaves.append(module.store_fold("fold_kernel", next(values), track))
            if module.direct.expand.bias is not None:
                leaves.append(module.store_fold("fold_bias", next(values), track))
        del values
        # A backward reaching any leaf is the signal that the block was used.
        reached: list[bool] = []
        hooks = [
            leaf.register_post_accumulate_grad_hook(lambda _: reached.append(True))
            for leaf in leaves
            if track
        ]
        for module in modules:
            module.fold_active = True
        try:
            yield
        finally:
            for module in modules:
                module.fold_active = False
            for hook in hooks:
                hook.remove()
            grads = [leaf.grad for leaf in leaves]
            # Plain buffers outside a block: moving or copying the model stays safe.
            for leaf in leaves:
                leaf.grad = None
                leaf.requires_grad_(False)
        if not track:
            return
        if not reached:
            raise RuntimeError(
                "fold_cache() closed without a backward: backpropagate the frames "
                "inside the block, or enter it under torch.no_grad()"
            )
        with torch.enable_grad():
            roots = [
                (value, grad)
                for value, grad in zip(composition(), grads, strict=True)
                if value.requires_grad
            ]
            torch.autograd.backward(
                [value for value, _ in roots], [grad for _, grad in roots]
            )

    def prepare_for_compile(
        self,
        input_shape: tuple[int, ...] | None = None,
        input_dtype: torch.dtype | None = None,
    ) -> None:
        """Select training execution paths without changing the mathematics.

        input_shape is the B,T,3,H,W clip shape and input_dtype the autocast
        dtype. BF16 or FP16 training selects the bundled direct local-attention
        kernels where Triton is available. BF16 training on CUDA folds context
        images into heads to share masks and, with attn_type "hybrid" outside
        deterministic mode, primes Flex masks for the trunk size of input_shape
        (with PU2 padding), shared across equal context windows. Without a
        usable input_shape an auto-selected hybrid runs the masked path; an
        explicit one raises. Repeated calls replace caches; eval and export keep
        portable SDPA. Per-call device, dtype and geometry checks still govern
        execution.
        """
        if input_shape is not None and (len(input_shape) != 5 or input_shape[2] != 3):
            raise ValueError("RDUAL preparation requires B,T,3,H,W")
        training = self.training and not self.deployed
        bf16_context = training and input_dtype == torch.bfloat16
        direct_local = (
            training
            and _HAS_DIRECT_ATTENTION
            and input_dtype in (torch.bfloat16, torch.float16)
        )
        flex_shape = None
        if (
            bf16_context
            and self.attn_type == "hybrid"
            and self.mean.device.type == "cuda"
            and not torch.are_deterministic_algorithms_enabled()
        ):
            if input_shape is not None and all(
                type(d) is int and d >= 1 for d in input_shape
            ):
                height, width = input_shape[-2:]
                flex_shape = ((height + 3) // 4) * 2, ((width + 3) // 4) * 2
            elif self.require_hybrid:
                raise ValueError(
                    "Hybrid compilation requires a fixed positive B,T,3,H,W input_shape"
                )
        shared_masks: dict[tuple[int, int], Any] = {}
        for layer in self.modules():
            if not isinstance(layer, _SpatialAttention):
                continue
            masks: dict[tuple[int, int, int], Any] = {}
            layer.flex_masks = masks
            layer.na_training_fast = bool(
                bf16_context and layer.context and layer.window in (32, 64)
            )
            if flex_shape is not None and layer.na_training_fast:
                height, width = flex_shape
                if height > layer.window or width > layer.window:
                    count = ((height + 15) // 16) * ((width + 15) // 16)
                    for first in range(0, count, 4):
                        key = layer.window, first
                        if key not in shared_masks:
                            shared_masks[key] = _na_block_mask(
                                height, width, layer.window, first, self.mean.device
                            )
                        masks[(height, width, first)] = shared_masks[key]
            layer.g_attention_impl = (
                "direct" if direct_local and not layer.context else "sdpa"
            )

    def prepare_for_export(self) -> None:
        """Select portable SDPA and fold factors; call only on an eval COPY."""
        if self.training:
            raise RuntimeError("Call eval() on an export copy before folding RDUAL")
        for layer in self.modules():
            if isinstance(layer, _SpatialAttention):
                layer.g_attention_impl = "sdpa"
                layer.na_training_fast = False
                layer.flex_masks.clear()
        if self.deployed:
            return
        for name in self.reparameterized_names:
            parent, _, leaf = name.rpartition(".")
            module = self.get_submodule(name)
            setattr(self.get_submodule(parent), leaf, module.folded())
        self.deployed = True

    def _pad(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        ph, pw = (-height) % 4, (-width) % 4
        if ph or pw:
            mode = "reflect" if height > ph and width > pw else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return x

    def _prefix(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """State-independent work: stem, first group and coarse retrieval keys."""
        shifted = x - self.mean.to(x.dtype)
        stem = self.stem(F.pixel_unshuffle(shifted, 2))
        group = self.groups[0]
        check = self.use_checkpoint and self.training and torch.is_grad_enabled()
        h = fine = stem
        for index, layer in enumerate(group.layers):
            h = _checkpoint_layer(layer, h) if check else layer(h)
            if index == 1:
                fine = h
        h = stem + group.conv(h)
        coarse = self.stem(F.pixel_unshuffle(self.xr.downsample(shifted), 2))
        for layer in group.layers[:2]:
            coarse = _checkpoint_layer(layer, coarse) if check else layer(coarse)
        return shifted, stem, fine, h, self.xr.key(self.xr.norm_f(coarse))

    def _fuse(
        self,
        center: Tensor,
        states: list[Tensor],
        valid: list[Tensor],
        flow: list[Tensor],
        slots: list[int],
    ) -> Tensor:
        if torch.compiler.is_compiling():
            # A graph constant: no host-to-device copy at run time.
            index = torch.tensor(slots, device=center.device)
        else:
            # Slots are [past], [future] or [past, future]: a contiguous range, made
            # on the device instead of uploaded from the host on every call.
            index = torch.arange(slots[0], slots[0] + len(slots), device=center.device)
        args = (
            center,
            torch.stack(states, 1),
            torch.stack(valid, 1),
            torch.stack(flow, 1),
            index,
        )
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            context = nullcontext()
            if not torch.compiler.is_compiling() and torch.is_autocast_enabled(
                center.device.type
            ):
                context = torch.autocast(
                    center.device.type,
                    dtype=torch.get_autocast_dtype(center.device.type),
                    cache_enabled=False,
                )
            with context:
                return checkpoint(
                    self.temporal,
                    *args,
                    use_reentrant=False,
                    context_fn=partial(
                        create_selective_checkpoint_contexts,
                        _temporal_checkpoint_policy,
                    ),
                )
        return self.temporal(*args)

    def _trunk(
        self, h: Tensor, fine: Tensor, key: Tensor
    ) -> tuple[Tensor, Tensor, tuple]:
        """Everything after the fusion point up to the carried state."""
        h = self.groups[1](h)
        height, width = h.shape[-2:]
        u = self.xr.norm_u(pad_even(h))
        q = self.xr.query(torch.cat((self.xr.norm_f(pad_even(fine)), u), 1))
        packed = pack_qk(q, key)
        o = attend_packed(packed, pack_children(self.xr.value(u)), self.xr.log_tau)
        lr = self.xr.project(o * self.xr.gate(u).sigmoid())[:, :, :height, :width]
        hr = self.xr.hr(unpack_children(o))[:, :, : 2 * height, : 2 * width]
        h = h + lr
        for group in self.groups[2:]:
            h = group(h)
        return h, hr, packed

    def _tail(
        self, shifted: Tensor, stem: Tensor, h: Tensor, hr: Tensor, packed: tuple
    ) -> Tensor:
        height, width = h.shape[-2:]
        features = stem + self.trunk_end(self.final_norm(h))
        t2 = self.tail[:4](features) + hr
        out = self.tail[4:](t2)
        if self.internal_scale >= 4:
            extra = self.xg(t2, packed, (height, width))
            out = out + (
                F.pixel_shuffle(extra, 2) if self.internal_scale == 8 else extra
            )
        return (
            out
            + F.interpolate(shifted, scale_factor=self.scale, mode="nearest")
            + self.mean.to(shifted.dtype)
        )

    def encode(self, frames: Tensor) -> tuple[Tensor, ...]:
        """A frame's state-independent part: stem, first group, coarse keys.

        frames is B,3,H,W, padded here to a multiple of four. Returns (shifted
        input, stem, fine features, trunk features, coarse keys);
        _ENCODED_TRUNK names the trunk features' position. Each image is
        encoded on its own, so frames of several time steps can share one call
        and be split along the batch dimension afterwards.
        """
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("RDUAL step input must be B,3,H,W")
        return self._prefix(self._pad(frames))

    def step_encoded(
        self,
        encoded: tuple[Tensor, ...],
        state: Tensor | None = None,
        flow: Tensor | None = None,
        valid: Tensor | None = None,
        *,
        future: tuple[Tensor, Tensor, Tensor] | None = None,
        output: bool = True,
    ) -> tuple[Tensor | None, Tensor]:
        """One frame from the result of encode(), plus its donors.

        flow is B,5,Htrunk,Wtrunk mapping this frame onto the previous one,
        its x/y flow followed by that donor's alignment cues as pack_guidance()
        gives them, and valid is a boolean B. future is an optional (state,
        flow, valid) donor from a reverse sweep, its flow mapping this frame
        onto the next one. Returns the output at the padded frame size and the
        state to carry; with output=False the tail is skipped and the output
        is None.
        """
        shifted, stem, fine, h, key = encoded
        states, valids, flows, slots = [], [], [], []
        if state is not None:
            if flow is None or flow.requires_grad:
                raise ValueError("A carried state requires detached flow guidance")
            if valid is None:
                valid = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
            states.append(state)
            valids.append(valid)
            flows.append(flow)
            slots.append(_PAST)
        if future is not None:
            if future[1].requires_grad:
                raise ValueError("A future state requires detached flow guidance")
            states.append(future[0])
            flows.append(future[1])
            valids.append(future[2])
            slots.append(_FUTURE)
        if states:
            h = self._fuse(h, states, valids, flows, slots)
        h, hr, packed = self._trunk(h, fine, key)
        out = self._tail(shifted, stem, h, hr, packed) if output else None
        return out, h

    def _clip_edges(
        self,
        frames: Tensor,
        guidance: Tensor,
        frame_valid: Tensor | None = None,
        shot_ids: Tensor | None = None,
    ) -> Tensor:
        """Check a clip's inputs; returns the boolean B,T-1 temporal edges.

        Edge i links frames i and i+1: both valid and in the same shot.
        """
        if frames.ndim != 5 or frames.shape[1] < 1 or frames.shape[2] != 3:
            raise ValueError("RDUAL input must be B,T,3,H,W")
        batch, count, _, original_h, original_w = frames.shape
        expected = (
            batch,
            count - 1,
            2,
            2,
            ((original_h + 3) // 4) * 2,
            ((original_w + 3) // 4) * 2,
        )
        if guidance.shape != expected:
            raise ValueError("Guidance must match B,T-1,2,2,Htrunk,Wtrunk")
        if guidance.device != frames.device or guidance.requires_grad:
            raise ValueError("Guidance must be detached and on the frame device")
        # A non-finite flow would index outside the donor in the warp.
        if not torch.compiler.is_compiling() and not bool(
            torch.isfinite(guidance).all()
        ):
            raise ValueError("Guidance must be finite")
        if frame_valid is None:
            frame_valid = torch.ones(
                (batch, count), dtype=torch.bool, device=frames.device
            )
        if frame_valid.shape != (batch, count) or frame_valid.dtype != torch.bool:
            raise ValueError("frame_valid must be boolean B,T")
        if shot_ids is None:
            shot_ids = torch.zeros(
                (batch, count), dtype=torch.int64, device=frames.device
            )
        if shot_ids.shape != (batch, count):
            raise ValueError("shot_ids must be B,T")
        return (
            frame_valid[:, 1:]
            & frame_valid[:, :-1]
            & (shot_ids[:, 1:] == shot_ids[:, :-1])
        )

    def pack_guidance(self, frames: Tensor, guidance: Tensor) -> Tensor:
        """Guidance with every donor's alignment cues, B,T-1,2,5,Htrunk,Wtrunk.

        Each entry keeps its x/y flow and appends the cues of alignment_cues()
        on its target frame's trunk grid, computed without gradients from the
        frames padded as encode() pads them and 2x2 averaged to trunk size.
        Entry [i,0] samples frame i on frame i+1's grid with [i,1] as its
        reverse flow, and entry [i,1] samples frame i+1 on frame i's grid with
        [i,0] as its reverse. forward(), infer() and backward_clip() call this
        themselves; per-frame callers pass its entries as step_encoded() flows.
        """
        batch, count = frames.shape[:2]
        pairs, directions, _, height, width = guidance.shape[1:]
        precision = torch.float64 if guidance.dtype == torch.float64 else torch.float32
        with torch.no_grad(), torch.autocast(frames.device.type, enabled=False):
            small = F.avg_pool2d(self._pad(frames.flatten(0, 1)).to(precision), 2)
            small = small.reshape(batch, count, 3, height, width)
            earlier, later = small[:, :-1].flatten(0, 1), small[:, 1:].flatten(0, 1)
            flows = guidance.to(precision).flatten(0, 1)
            cues = (
                alignment_cues(later, earlier, flows[:, 0], flows[:, 1]),
                alignment_cues(earlier, later, flows[:, 1], flows[:, 0]),
            )
            packed = torch.stack(cues, 1).reshape(
                batch, pairs, directions, 3, height, width
            )
            return torch.cat((guidance, packed.to(guidance.dtype)), 3)

    def forward(
        self,
        frames: Tensor,
        guidance: Tensor | None = None,
        frame_valid: Tensor | None = None,
        shot_ids: Tensor | None = None,
    ) -> Tensor:
        """B,T,3,H,W frames in 0..1 to B,T,3,sH,sW.

        guidance None computes it with guidance(); a tensor is used as given.
        """
        if guidance is None:
            guidance = self.guidance(frames)
        edge = self._clip_edges(frames, guidance, frame_valid, shot_ids)
        original_h, original_w = frames.shape[-2:]
        outputs, _ = self._sweeps(
            frames, self.pack_guidance(frames, guidance), edge, None, frames.shape[1]
        )
        out = torch.stack(outputs, 1)
        return out[..., : original_h * self.scale, : original_w * self.scale]

    def _sweeps(
        self,
        frames: Tensor,
        guidance: Tensor,
        edge: Tensor,
        past: tuple[Tensor, Tensor, Tensor] | None,
        count: int,
    ) -> tuple[list[Tensor], Tensor | None]:
        """forward()'s sweeps over B,n,3,H,W frames, output for the first count.

        guidance is packed and edge is B,n-1, both linking frames t and t+1.
        past is the (state, flow, valid) donor of the first frame, carried
        from the frame before it, or None. The reverse sweep starts at the
        last frame with no future. Returns the padded outputs and the state
        carried from frame count-1.
        """
        n = frames.shape[1]
        prefix = [self._prefix(self._pad(frames[:, t])) for t in range(n)]
        future: dict[int, Tensor] = {}
        state = None
        # Frame 0's reverse state would have no reader: frame t uses frame t+1's.
        for t in reversed(range(1, n)):
            _, _, fine, h, key = prefix[t]
            if state is not None:
                h = self._fuse(h, [state], [edge[:, t]], [guidance[:, t, 1]], [_FUTURE])
            state = self._trunk(h, fine, key)[0]
            future[t] = state
        outputs = []
        state, first_flow, first_valid = (None, None, None) if past is None else past
        for t in range(count):
            shifted, stem, fine, h, key = prefix[t]
            states, valid, flow, slots = [], [], [], []
            if state is not None:
                states.append(state)
                valid.append(edge[:, t - 1] if t else first_valid)
                flow.append(guidance[:, t - 1, 0] if t else first_flow)
                slots.append(_PAST)
            if t + 1 < n:
                states.append(future[t + 1])
                valid.append(edge[:, t])
                flow.append(guidance[:, t, 1])
                slots.append(_FUTURE)
            if states:
                h = self._fuse(h, states, valid, flow, slots)
            h, hr, packed = self._trunk(h, fine, key)
            outputs.append(self._tail(shifted, stem, h, hr, packed))
            state = h
        return outputs, state

    @torch.no_grad()
    def infer(
        self,
        frames: Tensor,
        guidance: Tensor | None = None,
        frame_valid: Tensor | None = None,
        shot_ids: Tensor | None = None,
        *,
        chunk: int = 0,
        lookahead: int = 8,
        offload: torch.device | str | None = None,
    ) -> Tensor:
        """forward() without gradients in chunks of chunk frames, for long clips.

        The inputs may live on any device; each chunk moves only its frames,
        guidance and edges, plus its look-ahead and the frame before it, to the
        model's device. Outputs go to offload when given, else stay on the
        model's device; returns B,T,3,sH,sW. chunk <= 0 or chunk >= T gives
        forward()'s result. Edges and alignment cues are those forward() uses,
        wherever the chunk boundaries fall.

        The forward state is carried from chunk to chunk. A chunk [c0, c1)
        starts its reverse sweep at frame min(T, c1 + lookahead) - 1 with no
        future, so it pays lookahead extra state-only reverse steps; frames
        past c1 produce no output. The result equals
        forward() when chunk + lookahead >= T, where every chunk's sweep starts
        at the clip's end; otherwise frames whose truncated future reaches them
        through the reverse states, and every later frame through the carried
        state, differ from it. How much look-ahead is worth its cost is a
        property of the trained weights.

        guidance None computes it once for the whole clip with guidance(), in
        groups of frame pairs on the model's device, kept on the frames'
        device; a tensor is used as given.
        """
        if type(chunk) is not int or type(lookahead) is not int:
            raise ValueError("chunk and lookahead must be ints")
        if lookahead < 0:
            raise ValueError("lookahead must be non-negative")
        if guidance is None:
            guidance = self.guidance(frames)
        edge = self._clip_edges(frames, guidance, frame_valid, shot_ids)
        count, _, original_h, original_w = frames.shape[1:]
        if chunk <= 0 or chunk >= count:
            chunk = count
        device = self.mean.device
        outputs: list[Tensor] = []
        state = None
        for first in range(0, count, chunk):
            stop = min(count, first + chunk)
            # The frame before the chunk only links the carried state.
            start, end = max(first - 1, 0), min(count, stop + lookahead)
            window = frames[:, start:end].to(device)
            packed = self.pack_guidance(window, guidance[:, start : end - 1].to(device))
            links = edge[:, start : end - 1].to(device)
            past = None
            if state is not None:
                past = (state, packed[:, 0, 0], links[:, 0])
                window, packed, links = window[:, 1:], packed[:, 1:], links[:, 1:]
            out, state = self._sweeps(window, packed, links, past, stop - first)
            for padded in out:
                frame = padded[
                    ..., : original_h * self.scale, : original_w * self.scale
                ]
                outputs.append(frame if offload is None else frame.to(offload))
            del window, packed, links, out
        return torch.stack(outputs, 1)

    def step_functions(
        self, compile: Callable[[Callable], Callable] | None = None
    ) -> StepFunctions:
        """The callables backward_clip() runs, each wrapped by compile when given.

        compile is for example torch.compile or a functools.partial of it with a
        mode; None keeps every callable eager. Their flows are 5-channel, as
        step_encoded() takes them.
        """

        def full(
            e: Encoded,
            s: Tensor,
            f: Tensor,
            v: Tensor,
            d: Tensor,
            fa: Tensor,
            lv: Tensor,
        ) -> tuple[Tensor | None, Tensor]:
            return self.step_encoded(e, s, f, v, future=(d, fa, lv))

        def reverse(e: Encoded, d: Tensor, fa: Tensor, lv: Tensor) -> Tensor:
            return self.step_encoded(e, future=(d, fa, lv), output=False)[1]

        def forward_state(
            e: Encoded,
            s: Tensor,
            f: Tensor,
            v: Tensor,
            d: Tensor,
            fa: Tensor,
            lv: Tensor,
        ) -> Tensor:
            return self.step_encoded(e, s, f, v, future=(d, fa, lv), output=False)[1]

        if compile is None:
            return StepFunctions(self.encode, full, reverse, forward_state, None)
        return StepFunctions(
            compile(self.encode),
            compile(full),
            compile(reverse),
            compile(forward_state),
            compile(self.compose_kernels),
        )

    def backward_clip(
        self,
        frames: Tensor,
        guidance: Tensor | None,
        loss: Callable[[Tensor, int], Tensor],
        *,
        frame_valid: Tensor | None = None,
        shot_ids: Tensor | None = None,
        keep: str = "none",
        functions: StepFunctions | None = None,
        autocast: Callable[[], AbstractContextManager] = nullcontext,
        grad_chunk: int = 2,
        encode_chunk: int = 5,
    ) -> Tensor:
        """Run one clip and backpropagate the sum of its per-frame losses.

        Parameter and frame gradients equal those of summing loss(out[:, t], t)
        over self(frames, guidance, frame_valid, shot_ids) and calling backward,
        and accumulate into .grad. loss is called once per frame, inside
        autocast(), with that frame's B,3,sH,sW output as a detached leaf, so it
        must not rely on reaching the model through it. A frame's loss graph is
        released as soon as its gradient at the output is known, so held modes
        hold only the model's graphs. The whole clip runs inside fold_cache(), so
        the caller only steps the optimizer afterwards. Returns the detached
        outputs, B,T,3,sH,sW. guidance None computes it with guidance() first,
        in eager code before any step mark; a tensor is used as given.

        keep chooses what is held instead of recomputed: "none" (no-gradient
        sweeps, then one replay per frame and direction), "reverse" (the reverse
        sweep's graphs are held until their backward) or "both" (every graph is
        held; no no-gradient pass and no replay). Held graphs trade memory for
        speed.

        Frames are encoded without gradients in calls of encode_chunk frames and
        with gradients at the end in calls of grad_chunk frames; a clip length
        either does not divide gives one more call shape, which a compiled encode
        compiles separately. functions comes from step_functions() (eager when
        None).

        For CUDA frames the method marks its own CUDA-graph steps
        (torch.compiler.cudagraph_mark_step_begin(), a no-op without CUDA
        graphs): each kernel composition of fold_cache(), each no-gradient
        encode, reverse and forward-state call, each frame of the forward and
        reverse replays (forward, loss and both backwards together) and each
        gradient encode chunk. Everything that outlives a step is copied into
        ordinary memory, and every parameter that requires gradients gets a
        zero .grad before the clip, so one that receives nothing ends the clip
        with zeros instead of None. With keep="none" each gradient leaf (a
        frame's encoded features, a state, a donor) also gets a zero gradient
        just before the step that first writes it, which later steps add to in
        place; held modes leave leaf gradients to autograd inside their span
        and copy the encoded features' gradients out of it before the gradient
        encode. Under CUDA graphs ("reduce-overhead",
        "max-autotune") keep="none" is the supported mode: held modes mark only
        the steps outside the span their held graphs cover, which then runs as
        one step. A loss callable may run compiled functions, which belong to
        the frame's step, but must not keep their outputs beyond the call
        without copying them; the caller marks steps for its own compiled calls
        outside this method.
        """
        if keep not in ("none", "reverse", "both"):
            raise ValueError(f"keep must be none/reverse/both, not {keep!r}")
        if grad_chunk < 1 or encode_chunk < 1:
            raise ValueError("grad_chunk and encode_chunk must be positive")
        fn = self.step_functions() if functions is None else functions

        if guidance is None:
            # Eager, before the first CUDA-graph step mark
            guidance = self.guidance(frames)
        edge = self._clip_edges(frames, guidance, frame_valid, shot_ids)
        # Ordinary eager memory before the first step; the cues need no graph.
        guidance = self.pack_guidance(frames, guidance)
        batch, count, _, original_h, original_w = frames.shape
        height, width = original_h * self.scale, original_w * self.scale
        # A compiled step guards on strides, so every per-frame slice is contiguous.
        images = [frames[:, t].contiguous() for t in range(count)]
        closed_flow = guidance.new_zeros((batch, *guidance.shape[-3:]))
        closed = torch.zeros(batch, dtype=torch.bool, device=frames.device)
        # An absent donor is a zero state behind a closed edge, which masks it, so
        # every frame of a clip has the same call shape.
        past_flow = [closed_flow] + [
            guidance[:, t - 1, 0].contiguous() for t in range(1, count)
        ]
        past_valid = [closed] + [edge[:, t - 1].contiguous() for t in range(1, count)]
        ahead_flow = [guidance[:, t, 1].contiguous() for t in range(count - 1)] + [
            closed_flow
        ]
        ahead_valid = [edge[:, t].contiguous() for t in range(count - 1)] + [closed]
        outputs: dict[int, Tensor] = {}
        marks = _marks_cuda_graph_steps(frames)
        # Held graphs span the sweeps and replays, which then form one step.
        replay = keep == "none"

        def step(held: bool = False) -> None:
            if marks and (replay or not held):
                torch.compiler.cudagraph_mark_step_begin()

        # Steps marked per frame: leaf gradients written in one step are read
        # in later ones, so they live in ordinary memory from the start.
        per_frame = marks and replay
        if marks:
            # Gradients accumulate in place in ordinary memory, never adopted
            # from a step's output. Held modes need this too: the gradient
            # encode chunks and the closing composition add to parameter
            # gradients in steps of their own, after the held span's mark.
            for parameter in self.parameters():
                if parameter.requires_grad and parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
        composition = self.compose_kernels if fn.compose is None else fn.compose

        def compose() -> list[Tensor]:
            step()
            return composition()

        def leaf(x: Tensor) -> Tensor:
            """A gradient leaf, made right before the step that first writes it."""
            x = x.detach().requires_grad_()
            if per_frame:
                x.grad = torch.zeros_like(x)
            return x

        def buffer(features: Encoded) -> None:
            """Zero gradients for a frame's encoded leaves before its first step."""
            if per_frame:
                for x in features:
                    x.grad = torch.zeros_like(x)

        with self.fold_cache(compose):
            encoded: list[Encoded] = []
            with torch.no_grad(), autocast():
                for first in range(0, count, encode_chunk):
                    step()
                    span = images[first : first + encode_chunk]
                    together = fn.encode(torch.cat(span))
                    encoded += [
                        tuple(x[i * batch : (i + 1) * batch].clone() for x in together)
                        for i in range(len(span))
                    ]
                    # The batched copy would otherwise stay alive through the sweeps.
                    del together
            # The encoded features are leaves shared by every step that reads a
            # frame: its forward step and its reverse step.
            # Their gradients collect both adjoints, and one encode with
            # gradients at the end pushes the sum into the parameters and frames,
            # exact by linearity of the backward. Each frame's features get
            # their gradient buffers only when its first gradient step comes.
            encoded = [
                tuple(x.requires_grad_() for x in features) for features in encoded
            ]
            zero = torch.zeros_like(encoded[0][_ENCODED_TRUNK])
            future = [zero] * (count + 1)
            states = [zero]

            def ahead(t: int, donor: Tensor | None = None) -> tuple[Tensor, ...]:
                return (
                    future[t + 1] if donor is None else donor,
                    ahead_flow[t],
                    ahead_valid[t],
                )

            def frame(
                t: int, state: Tensor, donor: Tensor
            ) -> tuple[Tensor, Tensor, Tensor]:
                """The cropped output, the loss gradient there, the carried state.

                The loss backpropagates at once into a detached copy of the
                output, so its graph is gone before the model's is used.
                """
                with autocast():
                    output, carried = fn.full(
                        encoded[t], state, past_flow[t], past_valid[t], *ahead(t, donor)
                    )
                    assert output is not None
                    output = output[..., :height, :width]
                    target = output.detach().requires_grad_()
                    value = loss(target, t)
                value.backward()
                del value
                assert target.grad is not None
                outputs[t] = target.detach().clone()
                return output, target.grad, carried

            produced: list[Tensor | None] = [None] * (count + 1)
            leaves: list[Tensor | None] = [None] * (count + 1)
            if keep != "none":
                for t in reversed(range(1, count)):
                    leaves[t] = leaf(future[t + 1])
                    with autocast():
                        out = fn.reverse(
                            encoded[t], leaves[t], ahead_flow[t], ahead_valid[t]
                        )
                    produced[t], future[t] = out, out.detach()
                    del out
            else:
                with torch.no_grad(), autocast():
                    for t in reversed(range(1, count)):
                        step()
                        future[t] = fn.reverse(
                            _plain(encoded[t]),
                            future[t + 1],
                            ahead_flow[t],
                            ahead_valid[t],
                        ).clone()

            pending: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
            if keep == "both":
                for t in range(count):
                    state = leaf(states[t])
                    donor = leaf(future[t + 1])
                    output, grad, carried = frame(t, state, donor)
                    states.append(carried.detach())
                    pending.append((output, grad, carried, state, donor))
                    del output, grad, carried
            else:
                with torch.no_grad(), autocast():
                    for t in range(count - 1):
                        step(held=True)
                        states.append(
                            fn.forward_state(
                                _plain(encoded[t]),
                                states[t],
                                past_flow[t],
                                past_valid[t],
                                *ahead(t),
                            ).clone()
                        )
            # Backward runs from the last frame so each carried state's adjoint
            # is known before its producer's graph is built or released.
            adjoint: Tensor | None = None
            owed: list[Tensor | None] = [None] * (count + 1)
            for t in reversed(range(count)):
                if keep == "both":
                    output, grad, carried, state, donor = pending.pop()
                else:
                    buffer(encoded[t])
                    step(held=True)
                    state = leaf(states[t])
                    donor = leaf(future[t + 1])
                    output, grad, carried = frame(t, state, donor)
                roots, adjoints = [output], [grad]
                if adjoint is not None:
                    roots.append(carried)
                    adjoints.append(adjoint.to(carried.dtype))
                torch.autograd.backward(roots, adjoints)
                del output, grad, carried, roots, adjoints
                adjoint = state.grad
                owed[t + 1] = donor.grad
            # The reverse chain runs from frame 1 forwards: frame t's reverse
            # state feeds frame t-1's steps, so its adjoint is what the
            # forward replays owed it plus what frame t-1's reverse step
            # passed back.
            carry: Tensor | None = None
            for t in range(1, count):
                if keep == "none":
                    step()
                    donor = leaf(future[t + 1])
                    with autocast():
                        out = fn.reverse(
                            encoded[t], donor, ahead_flow[t], ahead_valid[t]
                        )
                else:
                    donor, out = leaves[t], produced[t]
                    produced[t] = leaves[t] = None
                owed_t = owed[t]
                assert donor is not None and out is not None
                assert owed_t is not None
                pull = owed_t if carry is None else owed_t + carry
                (out.to(pull.dtype) * pull).sum().backward()
                del out
                carry = donor.grad

            if marks and not replay:
                # Adopted inside the held span, read by the marked encode chunks:
                # into ordinary memory now that the held graphs are released.
                for features in encoded:
                    for x in features:
                        assert x.grad is not None
                        x.grad = x.grad.clone()
            # Every encoded feature is an input of its frame's full step, whose
            # output gradient is backpropagated in every mode, so each has one.
            # Encoding every frame with gradients in one call doubled peak memory;
            # grad_chunk frames per call, each chunk's features and adjoints
            # released before the next.
            for first in range(0, count, grad_chunk):
                step()
                span = encoded[first : first + grad_chunk]
                with autocast():
                    together = fn.encode(torch.cat(images[first : first + grad_chunk]))
                grads = [
                    torch.cat([features[k].grad for features in span])
                    for k in range(len(together))
                ]
                encoded[first : first + grad_chunk] = [()] * len(span)
                del span
                # The shifted input has no graph unless the frames need gradients.
                roots = [
                    (x, g)
                    for x, g in zip(together, grads, strict=True)
                    if x.requires_grad
                ]
                torch.autograd.backward([x for x, _ in roots], [g for _, g in roots])
                del together, grads, roots
        return torch.stack([outputs[t] for t in range(count)], 1)


def rdual(scale: int = 4, use_checkpoint: bool = True, **kwargs: Any) -> RDUAL:
    return RDUAL(scale=scale, use_checkpoint=use_checkpoint, **kwargs)

