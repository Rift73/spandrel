"""
Faster inference path for ESRGAN's RRDB blocks on ROCm.

Adapted from florentia-mysteria/spandrel-rocm (MIT).

A residual dense block concatenates its input with every previous conv output
before the next conv (64, 96, 128, 160 and 192 channels), copying everything it
already has again each time, and runs LeakyReLU and the `* 0.2 + x` residuals
as separate passes. PyTorch's elementwise ops on ROCm run at a fraction of the
memory bandwidth, so with the convs on fast Winograd kernels this is about a
third of ESRGAN's run time.

Here every dense block works in one buffer [x | x1 | x2 | x3 | x4]: each conv
reads a channel prefix of it (no copy for a batch of 1), a HIP kernel applies
the activation while writing the conv output into the next slot, and another
one does the residuals, writing the block output straight into the first slot
of the next block's buffer. The convs themselves are unchanged, and the kernels
round after every operation like the separate PyTorch ops, so the results are
bit-identical to the original.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from spandrel.__helpers.rocm import kernels
from spandrel.__helpers.rocm.common import plain_inference

logger = logging.getLogger("spandrel.rocm")

_failed = False

# attribute of an RRDB output (a channel prefix) holding the buffer it lives in
_BUFFER = "_spandrel_rrdb_buffer"

# per dense block: [(conv, LeakyReLU slope)] * 4 + [(conv5, 0.0)]
_Layers = list[tuple[nn.Conv2d, float]]


def _is_conv3x3(m: nn.Module, in_ch: int, out_ch: int) -> bool:
    return (
        isinstance(m, nn.Conv2d)
        and m.in_channels == in_ch
        and m.out_channels == out_ch
        and m.kernel_size == (3, 3)
        and m.stride == (1, 1)
        and m.padding == (1, 1)
        and m.dilation == (1, 1)
        and m.groups == 1
        and m.padding_mode == "zeros"
    )


def _rdb_layout(rdb: nn.Module) -> tuple[int, int, _Layers] | None:
    """(nf, gc, layers) of a plain dense block (conv + LeakyReLU), or None."""
    if getattr(rdb, "conv1x1", None) is not None:
        return None  # ESRGAN+ extra residual paths
    first = getattr(rdb, "conv1", None)
    if not isinstance(first, nn.Sequential) or len(first) != 2:
        return None
    if not isinstance(first[0], nn.Conv2d):
        return None
    nf, gc = first[0].in_channels, first[0].out_channels
    layers: _Layers = []
    for i, name in enumerate(("conv1", "conv2", "conv3", "conv4")):
        seq = getattr(rdb, name, None)
        if not isinstance(seq, nn.Sequential) or len(seq) != 2:
            return None
        conv, act = seq[0], seq[1]
        # LeakyReLU only: the kernel computes exactly what PyTorch computes
        if not isinstance(act, nn.LeakyReLU) or not _is_conv3x3(conv, nf + i * gc, gc):
            return None
        layers.append((conv, float(act.negative_slope)))
    conv5 = getattr(rdb, "conv5", None)
    if isinstance(conv5, nn.Sequential) and len(conv5) == 1:
        conv5 = conv5[0]
    if conv5 is None or not _is_conv3x3(conv5, nf + 4 * gc, nf):
        return None
    layers.append((conv5, 0.0))
    return nf, gc, layers


def _layouts(module: nn.Module) -> list[_Layers] | None:
    """The three dense blocks' layers if this path supports the RRDB, cached."""
    cached = module.__dict__.get("_spandrel_rrdb_layout")
    if cached is None:
        rdbs = [_rdb_layout(getattr(module, n)) for n in ("RDB1", "RDB2", "RDB3")]
        sizes = {(r[0], r[1]) for r in rdbs if r is not None}
        ok = all(r is not None for r in rdbs) and len(sizes) == 1
        cached = ([r[2] for r in rdbs if r is not None] if ok else None,)
        module.__dict__["_spandrel_rrdb_layout"] = cached
    return cached[0]


def _input_buffer(x: torch.Tensor, channels: int) -> torch.Tensor:
    """
    A (B, channels, H, W) buffer whose first channels hold x. When x is the
    output of the previous RRDB on this path, it already is the first slot of
    such a buffer (marked by rrdb_forward), which is reused: only its other
    slots are written. Any other tensor is copied, so a caller's own buffer is
    never written into. The mark is an attribute rather than x._base, which
    views made in inference mode do not have.
    """
    b, c, h, w = x.shape
    buf = getattr(x, _BUFFER, None)
    if (
        isinstance(buf, torch.Tensor)
        and buf.shape == (b, channels, h, w)
        and buf.dtype == x.dtype
        and buf.device == x.device
        and buf.is_contiguous()
        and x.data_ptr() == buf.data_ptr()
        and x.stride() == buf[:, :c].stride()
    ):
        return buf
    buf = torch.empty((b, channels, h, w), device=x.device, dtype=x.dtype)
    buf[:, :c].copy_(x)
    return buf


def rrdb_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """RRDB.forward on the HIP kernels, or None if this path cannot be used."""
    global _failed
    if (
        _failed
        or not kernels.hip_kernels_available()
        or not plain_inference(x, module)
        or not kernels.can_rrdb(x)
    ):
        return None
    layouts = _layouts(module)
    if layouts is None:
        return None
    first = layouts[0][0][0]
    nf, gc = first.in_channels, first.out_channels
    if x.shape[1] != nf or any(
        conv.weight.dtype != x.dtype or conv.weight.device != x.device
        for layers in layouts
        for conv, _ in layers
    ):
        return None  # the original code converts or reports the mismatch
    channels = nf + 4 * gc
    try:
        b, _, h, w = x.shape
        cur = _input_buffer(x, channels)
        for k, layers in enumerate(layouts):
            for i, (conv, slope) in enumerate(layers[:4]):
                y = conv(cur[:, : nf + i * gc])
                kernels.act_into(y, cur, nf + i * gc, slope)
            y5 = layers[4][0](cur)
            nxt = torch.empty((b, channels, h, w), device=x.device, dtype=x.dtype)
            if k < 2:
                # RDB: x5 * 0.2 + x
                kernels.residual_add(y5, cur[:, :nf], nxt[:, :nf], 0.2)
            else:
                # last RDB and the RRDB around it: (x5 * 0.2 + x) * 0.2 + x_rrdb
                kernels.residual_add(y5, cur[:, :nf], nxt[:, :nf], 0.2, x, 0.2)
            cur = nxt
        out = cur[:, :nf]
        setattr(out, _BUFFER, cur)
        return out
    except torch.cuda.OutOfMemoryError:
        raise
    except kernels.FAILURES as e:
        _failed = True
        logger.warning(
            "The HIP RRDB kernels failed; using the original RRDB code for the "
            "rest of this process. %s: %s",
            type(e).__name__,
            e,
        )
        logger.debug("HIP RRDB kernel failure", exc_info=True)
        return None
