"""WSR: one five-tier family with shared context and exemplar components.

Default and Compact use exemplar retrieval; UltraCompact, SuperUltraCompact
and Mobile use the context-only trunk. Factories retain the established math,
initialization, checkpoint keys and export hooks without wrapper modules.
The five canonical WSR factories are the presets; no legacy/DIS2 aliases.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spandrel.util import store_hyperparameters

from ...SPAN.__arch.span import Conv3XC


class WideResidualBlock(nn.Module):
    """ReLU and residual add can become dense-convolution epilogues in TRT."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.expand = Conv3XC(channels, hidden, gain1=2)
        self.project = Conv3XC(hidden, channels, gain1=2)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.project(F.relu(self.expand(x)))


class ContextGate(nn.Module):
    """Independent smooth gain/offset maps per channel, plus local self-gating."""

    def __init__(self, channels: int, context_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(context_channels, 2 * channels, 1)
        self.local_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        maps = F.interpolate(
            self.project(context),
            scale_factor=4.0,
            mode="bilinear",
            align_corners=False,
        )
        # Always crop dynamically: padded coarse cells need not fill a frame.
        gamma, beta = maps[..., : x.shape[-2], : x.shape[-1]].chunk(2, dim=1)
        return x * (2.0 * torch.sigmoid(gamma + self.local_scale * x)) + beta


@torch.no_grad()
def _fold_conv3xc(source: Conv3XC) -> nn.Conv2d:
    """Compose in FP64, including the bias-filled halo from input-first pad.

    Padding BEFORE the first biased 1x1 is intentional in original Conv3XC:
    its halo carries b1, so one spatially constant folded bias is exact even
    at borders. Do not move that padding after the first 1x1.
    """
    if source.stride != 1 or source.has_relu:
        raise ValueError("WideContextSR folding expects stride-one linear Conv3XC")
    first, spatial, last = source.conv
    w1 = first.weight.detach().double()[..., 0, 0]
    w2 = spatial.weight.detach().double()
    w3 = last.weight.detach().double()[..., 0, 0]
    b1 = first.bias.detach().double()
    b2 = spatial.bias.detach().double()
    b3 = last.bias.detach().double()
    weight = torch.einsum("om,mnxy,ni->oixy", w3, w2, w1)
    weight = weight + F.pad(source.sk.weight.detach().double(), (1, 1, 1, 1))
    bias = w3 @ (w2.sum(dim=(2, 3)) @ b1 + b2) + b3
    bias = bias + source.sk.bias.detach().double()
    conv = nn.Conv2d(
        first.in_channels,
        last.out_channels,
        3,
        padding=1,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    conv.weight.copy_(weight)
    conv.bias.copy_(bias)
    return conv.eval().requires_grad_(False)


@store_hyperparameters()
class WideContextSR(nn.Module):
    """RGB x1/x2/x4 restoration with a wide trunk and an LR/4 context stream.

    Zero-initialized context projections start exactly at the ungated wide
    trunk. They are learnable: no warmup, schedule, auxiliary loss or teacher
    is involved. Coarse-stream gradients start flowing after those projections
    leave zero. Deployment tests must also exercise nonzero context weights.
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        width: int = 64,
        hidden: int = 128,
        depth: int = 8,
        sft_count: int = 3,
        context_width: int = 64,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if (num_in_ch, num_out_ch) != (3, 3) or scale not in (1, 2, 4):
            raise ValueError(
                "WideContextSR requires RGB input/output and scale in (1, 2, 4)"
            )
        if any(c < 8 or c % 8 for c in (width, hidden, context_width)):
            raise ValueError("All channel widths must be positive multiples of eight")
        if hidden < width or depth < 2 or not 1 <= sft_count <= depth:
            raise ValueError(
                "Require hidden >= width, depth >= 2 and 1 <= sft_count <= depth"
            )
        self.width = width
        self.hidden = hidden
        self.depth = depth
        self.sft_count = sft_count
        self.context_width = context_width
        self.scale = scale
        self.deploy = False
        self.merge_position = depth // 2 + 1
        positions = tuple(
            (j * depth + sft_count - 1) // sft_count for j in range(1, sft_count + 1)
        )
        self.gate_positions = positions
        self.context_indices = tuple(
            (4 * p + depth - 1) // depth - 1 for p in positions
        )
        if len(set(positions)) != sft_count or any(
            q > 1 and p < self.merge_position
            for p, q in zip(positions, self.context_indices, strict=True)
        ):
            raise ValueError("Context schedule attempts to consume a future merge")
        self.gate_at = {
            p: (j, q)
            for j, (p, q) in enumerate(
                zip(positions, self.context_indices, strict=True)
            )
        }

        self.stem = Conv3XC(3, width, gain1=2)
        self.blocks = nn.ModuleList(
            WideResidualBlock(width, hidden) for _ in range(depth)
        )
        self.context_stem = Conv3XC(48, context_width, gain1=2)
        self.context_blocks = nn.ModuleList(
            WideResidualBlock(context_width, context_width) for _ in range(4)
        )
        self.merge = nn.Conv2d(width, context_width, 4, stride=4)
        nn.init.zeros_(self.merge.weight)
        nn.init.zeros_(self.merge.bias)
        self.gates = nn.ModuleList(ContextGate(width, context_width) for _ in positions)
        self.deep = Conv3XC(width, width, gain1=2)
        self.reconstruct = nn.Conv2d(width, 3 * scale**2, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        padding = (0, (-width) % 4, 0, (-height) % 4)
        shallow = self.stem(x)
        context = F.relu(self.context_stem(F.pixel_unshuffle(F.pad(x, padding), 4)))
        c1 = self.context_blocks[0](context)
        c2 = self.context_blocks[1](c1)
        contexts = [c1, c2]
        current = shallow
        for position, block in enumerate(self.blocks, start=1):
            current = block(current)
            # Independent of gate positions: D8/S4 merges at block5 (no gate).
            if position == self.merge_position:
                c3 = self.context_blocks[2](c2 + self.merge(F.pad(current, padding)))
                c4 = self.context_blocks[3](c3)
                contexts.extend((c3, c4))
            if position in self.gate_at:
                gate_index, context_index = self.gate_at[position]
                current = self.gates[gate_index](current, contexts[context_index])
        return F.pixel_shuffle(
            self.reconstruct(self.deep(current) + shallow), self.scale
        )

    @torch.no_grad()
    def prepare_for_export(self) -> WideContextSR:
        """Permanently fold a separate eval copy; training checkpoints stay unfused."""
        if self.training:
            raise RuntimeError("Call eval() before prepare_for_export()")
        if self.deploy:
            return self

        def fold_children(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, Conv3XC):
                    setattr(module, name, _fold_conv3xc(child))
                else:
                    fold_children(child)

        fold_children(self)
        self.deploy = True
        return self


class ValidConv3XC(Conv3XC):
    """Gain-two Conv3XC on a valid spatial domain, including its cropped skip."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(cin, cout, gain1=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + self.sk(x[..., 1:-1, 1:-1])


class FoldableRGB5(nn.Module):
    """Two valid linear 3x3 branches with one initial pad; exact 5x5 self-fold."""

    def __init__(self, cout: int, middle: int = 64) -> None:
        super().__init__()
        self.first = ValidConv3XC(3, middle)
        self.second = ValidConv3XC(middle, cout)

    def forward(self, x: Tensor) -> Tensor:
        return self.second(self.first(F.pad(x, (2, 2, 2, 2))))

    @torch.no_grad()
    def folded(self) -> nn.Conv2d:
        if self.training:
            raise RuntimeError("Fold an eval copy only")
        # Compose already-collapsed branch parameters entirely in FP64.
        dtype = self.first.conv[0].weight.dtype
        first = _fold_conv3xc(copy.deepcopy(self.first).double())
        second = _fold_conv3xc(copy.deepcopy(self.second).double())
        weight = first.weight.new_zeros(second.out_channels, 3, 5, 5)
        for y in range(3):
            for x in range(3):
                weight[..., y : y + 3, x : x + 3] += torch.einsum(
                    "om,miyx->oiyx", second.weight[..., y, x], first.weight
                )
        bias = second.weight.sum((2, 3)) @ first.bias + second.bias
        result = nn.Conv2d(3, second.out_channels, 5, padding=2).to(
            device=weight.device, dtype=dtype
        )
        result.weight.copy_(weight)
        result.bias.copy_(bias)
        return result.eval().requires_grad_(False)


class BlockTokenAttention(nn.Module):
    """Fine queries retrieve compressed 4x4 LR block values within fixed windows.

    One packed mask coordinate per head leaves head_dim-1 cosine coordinates. The training
    path uses at least FP32 scores/softmax; the explicit deployment path uses the
    model dtype so a strongly-typed FP16 export can retain fused attention.
    """

    def __init__(self, channels: int = 64, window: int = 64, *, heads: int = 2) -> None:
        super().__init__()
        if (
            channels < 8
            or heads < 1
            or channels % heads
            or (channels // heads) % 8
            or window not in (32, 64)
        ):
            raise ValueError(
                "Require positive heads, channels/heads a positive multiple of eight, "
                "and window32 or64"
            )
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.cosine_dim = self.head_dim - 1
        self.query_channels = channels - heads
        self.window = window
        self.query = nn.Conv2d(channels, self.query_channels, 1)
        self.key_value = nn.Conv2d(
            channels, self.query_channels + channels, 4, stride=4
        )
        self.logit_scale = nn.Parameter(torch.full((heads,), math.log(10)))
        self.deploy = False
        self.register_buffer("export_scale", torch.ones(heads), persistent=False)
        with torch.no_grad():
            self.key_value.weight[self.query_channels :].zero_()
            self.key_value.bias[self.query_channels :].zero_()

    @staticmethod
    def _windows(x: Tensor, size: int) -> Tensor:
        batch, heads, dim, height, width = x.shape
        return (
            x.reshape(batch, heads, dim, height // size, size, width // size, size)
            .permute(0, 3, 5, 1, 4, 6, 2)
            .reshape(-1, heads, size * size, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        out, hp, wp = self.window_output(x)
        return self.restore(out, x.shape[0], hp, wp)[..., : x.shape[-2], : x.shape[-1]]

    def window_output(self, x: Tensor) -> tuple[Tensor, int, int]:
        batch, _, height, width = x.shape
        coarse_window = self.window // 4
        q = self.query(x).reshape(batch, self.heads, self.cosine_dim, height, width)
        kv = self.key_value(F.pad(x, (0, (-width) % 4, 0, (-height) % 4)))
        ch, cw = kv.shape[-2:]
        k, v = kv.split((self.query_channels, self.channels), dim=1)
        k = k.reshape(batch, self.heads, self.cosine_dim, ch, cw)
        v = v.reshape(batch, self.heads, self.head_dim, ch, cw)
        q = F.normalize(q, dim=2, eps=1e-6)
        k = F.normalize(k, dim=2, eps=1e-6)
        scale = (
            (
                self.export_scale
                if self.deploy
                else self.logit_scale.clamp(max=math.log(100)).exp()
            )
            .to(q.dtype)
            .view(1, self.heads, 1, 1, 1)
        )
        q = torch.cat((q * scale, torch.ones_like(q[:, :, :1])), dim=2)
        padding = (0, (-cw) % coarse_window, 0, (-ch) % coarse_window)
        # Every partial real block is valid; only extra window cells are masked.
        mask = F.pad(torch.zeros_like(k[:, :, :1]), padding, value=-10000.0)
        k = torch.cat((F.pad(k, padding), mask), dim=2)
        v = F.pad(v, padding)
        hp, wp = k.shape[-2] * 4, k.shape[-1] * 4
        q = F.pad(q, (0, wp - width, 0, hp - height))
        q = self._windows(q, self.window)
        k = self._windows(k, coarse_window)
        v = self._windows(v, coarse_window)
        return self._attention(q, k, v), hp, wp

    def restore(self, out: Tensor, batch: int, hp: int, wp: int) -> Tensor:
        return (
            out.reshape(
                batch,
                hp // self.window,
                wp // self.window,
                self.heads,
                self.window,
                self.window,
                self.head_dim,
            )
            .permute(0, 3, 6, 1, 4, 2, 5)
            .reshape(batch, self.channels, hp, wp)
        )

    def _attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.deploy:
            return (q @ k.transpose(-2, -1)).softmax(dim=-1) @ v
        score_dtype = torch.promote_types(q.dtype, torch.float32)
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = q.to(score_dtype) @ k.to(score_dtype).transpose(-2, -1)
            probabilities = scores.softmax(dim=-1)
        return probabilities.to(v.dtype) @ v

    @torch.no_grad()
    def prepare_for_export(self) -> BlockTokenAttention:
        if self.training:
            raise RuntimeError("Export an eval copy only")
        self.export_scale.copy_(self.logit_scale.clamp(max=math.log(100)).exp())
        self.deploy = True
        return self


@store_hyperparameters()
class WESR(nn.Module):
    """RGB x1/x2/x4 Wide Exemplar SR; the default retains D7/S3/window64.

    The full-resolution C64/H128 trunk receives smooth LR/4 context gates and
    one content-dependent retrieval from learned 4x4 block tokens. ``use_xsa``
    exposes a no-attention attribution twin, not a separately registered model.
    FE/Base are exact self-folds; removing WideSR's deep readout and changing
    its border semantics are intentional architecture changes, not exact rewrites.
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        depth: int = 7,
        gates: int = 3,
        window: int = 64,
        use_xsa: bool = True,
        width: int = 64,
        hidden: int = 128,
        context_width: int = 64,
        heads: int = 2,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if (num_in_ch, num_out_ch) != (3, 3) or scale not in (1, 2, 4):
            raise ValueError("WESR requires RGB input/output and scale in (1, 2, 4)")
        if any(c < 8 or c % 8 for c in (width, hidden, context_width)):
            raise ValueError("All channel widths must be positive multiples of eight")
        if hidden < width:
            raise ValueError("Require hidden >= width")
        if heads < 1 or width % heads or (width // heads) % 8:
            raise ValueError(
                "Require positive heads and width/heads a multiple of eight"
            )
        if depth < 3 or not 1 <= gates <= depth or window not in (32, 64):
            raise ValueError("Require depth>=3, 1<=gates<=depth and window32 or64")
        self.depth = depth
        self.width = width
        self.hidden = hidden
        self.context_width = context_width
        self.heads = heads
        self.gate_count = gates
        self.window = window
        self.scale = scale
        self.deploy = False
        self.merge_position = depth // 2 + 1
        self.gate_positions = tuple(
            (j * depth + gates - 1) // gates for j in range(1, gates + 1)
        )
        self.context_indices = tuple(
            (4 * p + depth - 1) // depth - 1 for p in self.gate_positions
        )
        if any(
            q > 1 and p < self.merge_position
            for p, q in zip(self.gate_positions, self.context_indices, strict=True)
        ):
            raise ValueError("Context schedule attempts to consume a future merge")
        eligible = [p for p in self.gate_positions if p <= depth - 2]
        if use_xsa and not eligible:
            raise ValueError("XSA needs a gate at least two blocks before the tail")
        self.xsa_position = max(eligible) if use_xsa else None
        self.gate_at = {
            p: (j, q)
            for j, (p, q) in enumerate(
                zip(self.gate_positions, self.context_indices, strict=True)
            )
        }
        self.stem = Conv3XC(3, width, gain1=2)
        self.first_expand = FoldableRGB5(hidden, middle=width)
        self.first_project = Conv3XC(hidden, width, gain1=2)
        self.blocks = nn.ModuleList(
            WideResidualBlock(width, hidden) for _ in range(depth - 1)
        )
        self.context_stem = Conv3XC(48, context_width, gain1=2)
        self.context_blocks = nn.ModuleList(
            WideResidualBlock(context_width, context_width) for _ in range(4)
        )
        self.merge = nn.Conv2d(width, context_width, 4, stride=4)
        nn.init.zeros_(self.merge.weight)
        nn.init.zeros_(self.merge.bias)
        self.gates = nn.ModuleList(
            ContextGate(width, context_width) for _ in self.gate_positions
        )
        self.reconstruct = nn.Conv2d(width, 3 * scale**2, 3, padding=1)
        self.base = FoldableRGB5(3 * scale**2, middle=width)
        # Create attention last so shared initializer RNG agrees with the twin.
        self.attention = (
            BlockTokenAttention(channels=width, heads=heads, window=window)
            if use_xsa
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        padding = (0, (-width) % 4, 0, (-height) % 4)
        current = self.stem(x) + self.first_project(F.relu(self.first_expand(x)))
        context = F.relu(self.context_stem(F.pixel_unshuffle(F.pad(x, padding), 4)))
        c1 = self.context_blocks[0](context)
        c2 = self.context_blocks[1](c1)
        contexts = [c1, c2]
        for position in range(1, self.depth + 1):
            if position > 1:
                current = self.blocks[position - 2](current)
            # Merge sees the pre-attention/pre-gate features at this position.
            if position == self.merge_position:
                c3 = self.context_blocks[2](c2 + self.merge(F.pad(current, padding)))
                c4 = self.context_blocks[3](c3)
                contexts.extend((c3, c4))
            if position == self.xsa_position and self.attention is not None:
                current = current + self.attention(current)
            if position in self.gate_at:
                j, q = self.gate_at[position]
                current = self.gates[j](current, contexts[q])
        return F.pixel_shuffle(self.reconstruct(current) + self.base(x), self.scale)

    @torch.no_grad()
    def prepare_for_export(self) -> WESR:
        """Permanently fold an eval copy; never fold the training instance."""
        if self.training:
            raise RuntimeError("Call eval() before prepare_for_export()")
        if self.deploy:
            return self

        def fold_children(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                # The whole valid chain must fold before generic Conv3XC children.
                if isinstance(child, FoldableRGB5):
                    setattr(module, name, child.folded())
                elif isinstance(child, BlockTokenAttention):
                    child.prepare_for_export()
                elif isinstance(child, Conv3XC):
                    setattr(module, name, _fold_conv3xc(child))
                else:
                    fold_children(child)

        fold_children(self)
        self.deploy = True
        return self


def wsr(num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4) -> WESR:
    """Default: unchanged WESR C64/H128/D7/S3, SPAN-F64 latency tier."""
    return WESR(num_in_ch=num_in_ch, num_out_ch=num_out_ch, scale=scale)


def wsr_compact(num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4) -> WESR:
    """Compact: unchanged WESR C32/H64/D9/S3 with exemplar attention."""
    return WESR(
        width=32,
        hidden=64,
        depth=9,
        gates=3,
        context_width=64,
        heads=1,
        window=64,
        use_xsa=True,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
    )


def wsr_ultracompact(
    num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4
) -> WideContextSR:
    """UltraCompact: unchanged context-only WSR C32/H64/D5/S2."""
    return WideContextSR(
        width=32,
        hidden=64,
        depth=5,
        sft_count=2,
        context_width=64,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
    )


def wsr_superultracompact(
    num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4
) -> WideContextSR:
    """SuperUltraCompact: unchanged context-only WSR C16/H32/D5/S2."""
    return WideContextSR(
        width=16,
        hidden=32,
        depth=5,
        sft_count=2,
        context_width=32,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
    )


def wsr_mobile(
    num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4
) -> WideContextSR:
    """Mobile: previously measured WSR C8/H16/D2/S1/context32 shortlist.

    Keeps the four-stage context stream and midpoint merge. Packaging this
    existing candidate does not constitute fresh all-scale latency calibration.
    """
    return WideContextSR(
        width=8,
        hidden=16,
        depth=2,
        sft_count=1,
        context_width=32,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
    )
