"""RSR: recurrent WSR Compact for video super-resolution.

RSR is WSR Compact (C32/H64/D9/S3, one Halo head, context width 64) whose Halo
attention stage additionally reads the trunk state of neighbouring frames.
There is no optical flow: a neighbour's state, the trunk after the last gate,
is tokenized exactly like the frame's own features and read by the same pixel
queries, with a separate softmax that contains a learned null token.

A clip (B,T,3,H,W) is processed in two sweeps. Every frame is encoded once up
to the Halo stage. A reverse sweep from frame T-1 down to frame 1 carries the
state backwards; a forward sweep from frame 0 then reads the forward state of
the previous frame as "past" and the reverse state of the next frame as
"future". Only the forward sweep produces images, (B,T,3,scale*H,scale*W).

Temporal values and the null token start at zero, so the temporal read adds
exactly zero: a WSR Compact checkpoint loads non-strictly and RSR then equals
WSR Compact per frame until training opens the values. A 4-D input is one
frame without neighbours and is identical to WSR Compact.

temporal_read="aligned" replaces the block read: the Halo stage does the
frame's own read only, then reads each neighbour's bounded state tanh(state)
at full resolution, moved into place by optical flow from a frozen NeuFlow v2
(flow.*, saved with the checkpoint, never trained) and weighted by a
forward-backward consistency confidence. One zero-initialized 3x3 blend per
sweep adds the reads to the trunk, so an aligned model also starts equal to
WSR Compact. Guidance (B,T,2,3,h,w) holds per frame, for the past [:, :, 0]
and the future [:, :, 1] neighbour, the flow in pixels of the flow grid h,w
and the inconsistency in frame pixels; guidance() computes it.

Keep training checkpoints unfolded. For deployment, make a separate copy,
call eval(), then prepare_for_export(). This conversion is in-place and is
not reversible.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from spandrel.util import store_hyperparameters

# Linear projections and residual/context components.


def _composed_conv(
    x: Tensor,
    module: Conv3XC | FoldableRGB5,
    dtype: torch.dtype,
    stride: int,
    padding: int,
) -> Tensor:
    """Convolve once with a kernel composed from branch parameters by autograd.

    Composition runs in FP32 (FP64 for double parameters) outside autocast and
    is cast back to the parameter dtype; autocast then handles the convolution
    as it would an ordinary layer. The kernel follows the input's memory format
    so cuDNN does not convert it on every call.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        weight, bias = module.composed(torch.promote_types(dtype, torch.float32))
    layout = (
        torch.channels_last
        if x.is_contiguous(memory_format=torch.channels_last)
        else torch.contiguous_format
    )
    weight = weight.to(dtype).contiguous(memory_format=layout)
    return F.conv2d(x, weight, bias.to(dtype), stride=stride, padding=padding)


class Conv3XC(nn.Module):
    """Reparameterizable 1x1 -> 3x3 -> 1x1 projection with a 1x1 skip.

    Training composes both branches into one 3x3 kernel on every forward, with
    gradients reaching the branch factors through the composition; this equals
    padding before the first biased projection and evaluating the branches
    explicitly. Evaluation refreshes a cached 3x3 convolution from those
    parameters. WSR's export preparation instead folds a separate eval copy
    permanently; the training branches remain the checkpoint source of truth.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        gain1: int = 1,
        gain2: int = 0,
        s: int = 1,
        bias: Literal[True] = True,
        relu: bool = False,
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )

        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False  # pyright: ignore[reportOptionalMemberAccess]
        self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        b1 = self.conv[0].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w2 = self.conv[1].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        b2 = self.conv[1].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]
        w3 = self.conv[2].weight.data.clone().detach()  # pyright: ignore[reportCallIssue]
        b3 = self.conv[2].bias.data.clone().detach()  # pyright: ignore[reportCallIssue]

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()  # pyright: ignore[reportOptionalMemberAccess]
        target_kernel_size = 3

        h_pixels_to_pad = (target_kernel_size - 1) // 2
        w_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [h_pixels_to_pad, h_pixels_to_pad, w_pixels_to_pad, w_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat.contiguous()
        self.eval_conv.bias.data = self.bias_concat.contiguous()  # pyright: ignore[reportOptionalMemberAccess]

    def composed(self, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Compose the branches into one 3x3 kernel and bias, keeping autograd.

        The bias includes the first projection's bias read through the 3x3
        taps, which the explicit branches see on the padded border as well.
        """
        first, spatial, last = self.conv
        w1 = first.weight.to(dtype).flatten(1)
        w2 = spatial.weight.to(dtype)
        w3 = last.weight.to(dtype).flatten(1)
        mid, inner = w2.shape[:2]
        # Taps-major factors are views of channels-last weights; the product
        # (out, 3, 3, in) is the channels-last layout of the 3x3 kernel.
        taps = w2.permute(0, 2, 3, 1).reshape(mid * 9, inner) @ w1
        kernel = (w3 @ taps.reshape(mid, -1)).reshape(w3.shape[0], 9, -1)
        kernel[:, 4] += self.sk.weight.to(dtype).flatten(1)
        weight = kernel.unflatten(1, (3, 3)).permute(0, 3, 1, 2)
        bias = w3 @ (w2.sum(dim=(2, 3)) @ first.bias.to(dtype) + spatial.bias.to(dtype))
        bias = bias + last.bias.to(dtype) + self.sk.bias.to(dtype)
        return weight, bias

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            out = _composed_conv(x, self, self.sk.weight.dtype, self.stride, padding=1)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


@torch.no_grad()
def _fold_conv3xc(source: Conv3XC) -> nn.Conv2d:
    """Fold a linear Conv3XC into one 3x3 convolution, composing in FP64.

    Conv3XC pads the input before its first biased 1x1 projection. The padded
    features therefore contain that projection's bias; including it is necessary
    for the folded bias to remain correct at image borders. The final tensors
    are cast back to the source dtype. Folding does not imply bitwise equality
    between low-precision execution paths.
    """
    if source.stride != 1 or source.has_relu:
        raise ValueError("Folding expects stride-one linear Conv3XC")
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


class ValidConv3XC(Conv3XC):
    """Conv3XC without spatial padding; crop the skip to the valid 3x3 domain."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(cin, cout, gain1=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + self.sk(x[..., 1:-1, 1:-1])


class FoldableRGB5(nn.Module):
    """Linear RGB projection with a 5x5 effective kernel.

    Two valid 3x3 operators share one initial two-pixel pad. With no intermediate
    activation, their kernels and biases can be composed into one 5x5 operator.
    Training runs that composition with autograd on every forward.
    """

    def __init__(self, cout: int, middle: int = 64) -> None:
        super().__init__()
        self.first = ValidConv3XC(3, middle)
        self.second = ValidConv3XC(middle, cout)

    def composed(self, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Compose both valid operators into one 5x5 kernel and bias."""
        first, first_bias = self.first.composed(dtype)
        second, second_bias = self.second.composed(dtype)
        # Full (flipped) convolution of the two kernels over the middle channels.
        weight = F.conv2d(
            first.transpose(0, 1), second.flip(2, 3), padding=2
        ).transpose(0, 1)
        bias = second.sum((2, 3)) @ first_bias + second_bias
        return weight, bias

    def forward(self, x: Tensor) -> Tensor:
        # Only this module's own operators compose; a subclass may hold others.
        composable = isinstance(self.first, ValidConv3XC) and isinstance(
            self.second, ValidConv3XC
        )
        if self.training and composable:
            return _composed_conv(
                x, self, self.first.sk.weight.dtype, stride=1, padding=2
            )
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


class WideResidualBlock(nn.Module):
    """Width-expanded residual block: x + project(ReLU(expand(x))).

    Each linear Conv3XC uses gain-two training branches and folds to one dense
    3x3 convolution for deployment; the ReLU between projections is retained.
    """

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.expand = Conv3XC(channels, hidden, gain1=2)
        self.project = Conv3XC(hidden, channels, gain1=2)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.project(F.relu(self.expand(x)))


def _upsample4(x: Tensor, dim: int) -> Tensor:
    """4x bilinear upsampling (align_corners=False) along dim -1 or -2.

    Output 4i+p reads source i + (2p - 3)/8 with clamped edges, so each phase
    is a fixed blend of the previous, centre and next samples. Neighbours come
    from sliced concatenation; the compiler fuses the blend into pointwise
    kernels whose backward needs no scatter.
    """
    n = x.shape[dim]
    prev = torch.cat((x.narrow(dim, 0, 1), x.narrow(dim, 0, n - 1)), dim)
    after = torch.cat((x.narrow(dim, 1, n - 1), x.narrow(dim, n - 1, 1)), dim)
    phases = (
        0.375 * prev + 0.625 * x,
        0.125 * prev + 0.875 * x,
        0.875 * x + 0.125 * after,
        0.625 * x + 0.375 * after,
    )
    return torch.stack(phases, dim=dim).flatten(dim - 1, dim)


class ContextGate(nn.Module):
    """Modulate full-resolution features with an LR/4 context map.

    The gain is 2*sigmoid(gamma + local_scale*x), followed by an additive beta.
    Zero-initialized projections and local_scale make the initial gate identity.
    Cropping the 4x-interpolated map handles input sizes not divisible by four.
    Training upsamples with an exact separable blend instead of interpolate.
    """

    def __init__(self, channels: int, context_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(context_channels, 2 * channels, 1)
        self.local_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        maps = self.project(context)
        if self.training:
            maps = _upsample4(_upsample4(maps, -2), -1)
        else:
            maps = F.interpolate(
                maps, scale_factor=4.0, mode="bilinear", align_corners=False
            )
        # Always crop dynamically: padded coarse cells need not fill a frame.
        gamma, beta = maps[..., : x.shape[-2], : x.shape[-1]].chunk(2, dim=1)
        return x * (2.0 * torch.sigmoid(gamma + self.local_scale * x)) + beta


# Halo retrieval.


class BlockTokenAttention(nn.Module):
    """Halo attention with pixel queries and learned 4x4 block keys/values.

    Each 32x32 query tile reads a 16x16 grid of coarse tokens: a 64x64 feature
    region with a 16-pixel margin around the tile. Adjacent tiles share donors.
    Queries are not pooled. Keys and values use a dense stride-four projection.

    Per head, one coordinate encodes the padding mask and the remaining
    head_dim-1 coordinates encode cosine similarity. Partially filled real
    tokens remain valid; only out-of-image tokens receive the -10000 mask.
    Values start at zero, initially disabling the attention residual.
    """

    def __init__(self, channels: int = 64, *, heads: int = 2) -> None:
        super().__init__()
        if channels < 8 or heads < 1 or channels % heads or (channels // heads) % 8:
            raise ValueError(
                "Require positive heads and channels/heads a positive multiple of eight"
            )
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.cosine_dim = self.head_dim - 1
        self.query_channels = channels - heads
        self.window = 32
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
        """Pack tiles as (batch*rows*cols, heads, size**2, head_features)."""
        batch, heads, dim, height, width = x.shape
        return (
            x.reshape(batch, heads, dim, height // size, size, width // size, size)
            .permute(0, 3, 5, 1, 4, 6, 2)
            .reshape(-1, heads, size * size, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        out, hp, wp = self.window_output(x)
        return self.restore(out, x.shape[0], hp, wp)[..., : x.shape[-2], : x.shape[-1]]

    @staticmethod
    def _donors(x: Tensor, rows: int, cols: int) -> Tensor:
        """Pack four adjacent 8x8 token blocks into 256 donors per query tile."""
        batch, heads, dim, _, _ = x.shape
        cells = x.reshape(batch, heads, dim, rows + 1, 8, cols + 1, 8)
        # Order is quadrant then row/column within quadrant, shared by K/V/mask.
        pieces = [
            cells[:, :, :, dy : dy + rows, :, dx : dx + cols, :]
            .permute(0, 3, 5, 1, 4, 6, 2)
            .reshape(batch * rows * cols, heads, 64, dim)
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
        ]
        return torch.cat(pieces, dim=2)

    def window_output(self, x: Tensor) -> tuple[Tensor, int, int]:
        batch, _, height, width = x.shape
        # Pad queries to 32 and source tokens to 4; crop only after restoration.
        rows, cols = (height + 31) // 32, (width + 31) // 32
        hp, wp = rows * 32, cols * 32
        q = self.query(x).reshape(batch, self.heads, self.cosine_dim, height, width)
        kv = self.key_value(F.pad(x, (0, (-width) % 4, 0, (-height) % 4)))
        ch, cw = kv.shape[-2:]
        k, v = kv.split((self.query_channels, self.channels), dim=1)
        k = k.reshape(batch, self.heads, self.cosine_dim, ch, cw)
        v = v.reshape(batch, self.heads, self.head_dim, ch, cw)
        q = F.normalize(q, dim=2, eps=1e-6)
        k = F.normalize(k, dim=2, eps=1e-6)
        scale = (
            self.export_scale
            if self.deploy
            else self.logit_scale.clamp(max=math.log(100)).exp()
        )
        q = torch.cat(
            (
                q * scale.to(q.dtype).view(1, self.heads, 1, 1, 1),
                torch.ones_like(q[:, :, :1]),
            ),
            dim=2,
        )
        # Four coarse cells form the 16-pixel halo on each side of a query tile.
        padding = (4, 8 * (cols + 1) - cw - 4, 4, 8 * (rows + 1) - ch - 4)
        mask = F.pad(torch.zeros_like(k[:, :, :1]), padding, value=-10000.0)
        k = torch.cat((F.pad(k, padding), mask), dim=2)
        v = F.pad(v, padding)
        q = self._windows(F.pad(q, (0, wp - width, 0, hp - height)), 32)
        return (
            self._attention(
                q, self._donors(k, rows, cols), self._donors(v, rows, cols)
            ),
            hp,
            wp,
        )

    def restore(self, out: Tensor, batch: int, hp: int, wp: int) -> Tensor:
        """Undo query tiling; the caller then removes right/bottom padding."""
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
        # Export uses model dtype to allow fused low-precision attention.
        if self.deploy:
            return (q @ k.transpose(-2, -1)).softmax(dim=-1) @ v
        # Training keeps score accumulation and softmax at least FP32.
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


@torch.no_grad()
def _fold_modules(module: nn.Module) -> None:
    """Replace linear training branches in-place; fold RGB chains as a unit."""
    for name, child in list(module.named_children()):
        if isinstance(child, FoldableRGB5):
            setattr(module, name, child.folded())
        elif isinstance(child, BlockTokenAttention):
            child.prepare_for_export()
        elif isinstance(child, Conv3XC):
            setattr(module, name, _fold_conv3xc(child))
        else:
            _fold_modules(child)


# Attention-enabled backbone.


class WESR(nn.Module):
    """Attention-enabled WSR backbone for Default and Compact.

    A full-resolution wide trunk is conditioned by four LR/4 context blocks.
    At the scheduled retrieval stage, Halo attention adds content-selected
    donor values before context modulation. Two linear RGB paths provide a
    wide first-stage projection and an additive reconstruction base.

    Context merge, gate projections and attention values start at zero.
    These output projections learn first, opening gradient paths into the
    context features and attention matching without changing the initial base.
    """

    def __init__(
        self,
        *,
        depth: int = 7,
        gates: int = 3,
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
            raise ValueError("WSR requires RGB input/output and scale in (1, 2, 4)")
        if any(c < 8 or c % 8 for c in (width, hidden, context_width)):
            raise ValueError("All channel widths must be positive multiples of eight")
        if hidden < width:
            raise ValueError("Require hidden >= width")
        if heads < 1 or width % heads or (width // heads) % 8:
            raise ValueError(
                "Require positive heads and width/heads a multiple of eight"
            )
        if depth < 3 or not 1 <= gates <= depth:
            raise ValueError("Require depth>=3 and 1<=gates<=depth")
        self.depth = depth
        self.width = width
        self.hidden = hidden
        self.context_width = context_width
        self.heads = heads
        self.gate_count = gates
        self.window = 64  # Donor support; attention query tiles are 32 pixels.
        self.scale = scale
        self.deploy = False
        self.merge_position = depth // 2 + 1
        # Spread gates through the trunk. Context stages 3/4 require the merge.
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
        if not eligible:
            raise ValueError(
                "Attention needs a gate at least two blocks before the tail"
            )
        self.xsa_position = max(eligible)
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
        self.attention = BlockTokenAttention(channels=width, heads=heads)

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
            if position == self.xsa_position:
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

        _fold_modules(self)
        self.deploy = True
        return self


# Temporal Halo retrieval and the recurrent model.

# What encode() returns: (current, c3, c4).
Encoded = tuple[Tensor, ...]


class StepFunctions(NamedTuple):
    """The per-frame callables of a clip, eager or compiled.

    full(frame, encoded, past, past_valid, future, future_valid,
    past_guidance, future_guidance) returns (output, state);
    reverse(encoded, future, future_valid, future_guidance) and
    forward_state(encoded, past, past_valid, future, future_valid,
    past_guidance, future_guidance) return the state only. The guidance
    arguments are None for the block read.
    """

    encode: Callable[[Tensor], Encoded]
    full: Callable[..., tuple[Tensor, Tensor]]
    reverse: Callable[..., Tensor]
    forward_state: Callable[..., Tensor]


def _plain(encoded: Encoded) -> Encoded:
    return tuple(x.detach() for x in encoded)


def _marks_cuda_graph_steps(frames: Tensor) -> bool:
    """Whether backward_clip() marks CUDA-graph steps: CUDA frames only."""
    return frames.is_cuda


def _gradient(x: Tensor) -> Tensor:
    """The gradient an encoded leaf collected over the sweeps."""
    assert x.grad is not None
    return x.grad


# Inconsistency, in frame pixels, of a position whose flow leaves the frame
_OUT_OF_FRAME = 1000.0


def flow_working_factor(height: int, width: int) -> int:
    """The largest f in (1, 2, 4, 8) with min(height, width) / f >= 256, else 1."""
    short = min(height, width)
    return max((f for f in (1, 2, 4, 8) if short / f >= 256), default=1)


def _inconsistency(
    flow: Tensor, back: Tensor, scale_x: float, scale_y: float
) -> Tensor:
    """N,1,h,w forward-backward inconsistency |flow(p) + back(p + flow(p))|.

    flow and back are N,2,h,w flows in pixels of their grid, flow on the grid
    of the frame it maps from and back on the other; the result is in frame
    pixels (x times scale_x, y times scale_y). A position whose flow leaves
    the frame gets _OUT_OF_FRAME.
    """
    # RDUAL's module imports Triton and Flex attention: aligned read only.
    from ...RDUAL.__arch.rdual_arch import warp_image

    height, width = flow.shape[-2:]
    loop = flow + warp_image(back, flow)
    error = torch.hypot(loop[:, 0] * scale_x, loop[:, 1] * scale_y)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=flow.dtype, device=flow.device),
        torch.arange(width, dtype=flow.dtype, device=flow.device),
        indexing="ij",
    )
    x, y = x + flow[:, 0], y + flow[:, 1]
    # The frame covers [-0.5, size - 0.5] around its pixel centres.
    outside = (x < -0.5) | (x > width - 0.5) | (y < -0.5) | (y > height - 0.5)
    return error.masked_fill(outside, _OUT_OF_FRAME)[:, None]


class TemporalBlockTokenAttention(BlockTokenAttention):
    """Halo attention that also reads block tokens of neighbouring frame states.

    The frame's own 256 donors per query tile keep their own softmax. A past
    and a future state are tokenized like the frame's own features (same
    padding to four, out-of-image mask lane and donor packing) through their
    own stride-four projections, joined by one learned null token with a
    constant zero value, and normalized by a second softmax. Both softmaxes
    share one query tensor; their outputs are summed before one restore.

    Neighbour values and the null token start at zero, so the temporal read is
    initially exactly zero and the pretrained spatial softmax is never
    diluted. An absent neighbour (None) contributes no tokens; a per-sample
    False in its valid mask masks all of its tokens for that sample instead.
    With every neighbour absent or masked, the null token takes all of the
    temporal probability and the temporal read is exactly zero.

    Neighbour values are temporal_gain * tanh(projection), so each temporal
    read is bounded per channel by |temporal_gain| whatever the magnitude of
    the neighbour state: a recurrent state cannot feed back without limit.
    """

    def __init__(self, channels: int = 64, *, heads: int = 2) -> None:
        super().__init__(channels, heads=heads)
        self.key_value_past = nn.Conv2d(
            channels, self.query_channels + channels, 4, stride=4
        )
        self.key_value_future = nn.Conv2d(
            channels, self.query_channels + channels, 4, stride=4
        )
        self.null_key = nn.Parameter(torch.zeros(heads, self.cosine_dim))
        self.null_bias = nn.Parameter(torch.zeros(heads))
        self.temporal_gain = nn.Parameter(torch.ones(channels))
        with torch.no_grad():
            for projection in (self.key_value_past, self.key_value_future):
                projection.weight[self.query_channels :].zero_()
                projection.bias[self.query_channels :].zero_()

    def _queries(self, x: Tensor, rows: int, cols: int) -> Tensor:
        """Scaled cosine queries plus the mask lane, packed per 32x32 tile."""
        batch, _, height, width = x.shape
        q = self.query(x).reshape(batch, self.heads, self.cosine_dim, height, width)
        q = F.normalize(q, dim=2, eps=1e-6)
        scale = (
            self.export_scale
            if self.deploy
            else self.logit_scale.clamp(max=math.log(100)).exp()
        )
        q = torch.cat(
            (
                q * scale.to(q.dtype).view(1, self.heads, 1, 1, 1),
                torch.ones_like(q[:, :, :1]),
            ),
            dim=2,
        )
        return self._windows(
            F.pad(q, (0, cols * 32 - width, 0, rows * 32 - height)), 32
        )

    def _tokens(
        self,
        projection: nn.Conv2d,
        x: Tensor,
        rows: int,
        cols: int,
        valid: Tensor | None = None,
        gain: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Pack 256 key/value donors per query tile, as the frame's own Halo read.

        gain (channels,), given for neighbours only, bounds the values to
        gain * tanh(value) before padding.
        """
        batch, _, height, width = x.shape
        kv = projection(F.pad(x, (0, (-width) % 4, 0, (-height) % 4)))
        ch, cw = kv.shape[-2:]
        k, v = kv.split((self.query_channels, self.channels), dim=1)
        k = k.reshape(batch, self.heads, self.cosine_dim, ch, cw)
        v = v.reshape(batch, self.heads, self.head_dim, ch, cw)
        if gain is not None:
            gain = gain.to(v.dtype).view(1, self.heads, self.head_dim, 1, 1)
            v = gain * torch.tanh(v)
        k = F.normalize(k, dim=2, eps=1e-6)
        padding = (4, 8 * (cols + 1) - cw - 4, 4, 8 * (rows + 1) - ch - 4)
        mask = F.pad(torch.zeros_like(k[:, :, :1]), padding, value=-10000.0)
        if valid is not None:
            # A sample without this neighbour masks every one of its tokens.
            absent = ~valid.to(device=mask.device, dtype=torch.bool)
            mask = mask.masked_fill(absent.view(batch, 1, 1, 1, 1), -10000.0)
        k = torch.cat((F.pad(k, padding), mask), dim=2)
        v = F.pad(v, padding)
        return self._donors(k, rows, cols), self._donors(v, rows, cols)

    def forward(
        self,
        x: Tensor,
        past: Tensor | None = None,
        future: Tensor | None = None,
        past_valid: Tensor | None = None,
        future_valid: Tensor | None = None,
    ) -> Tensor:
        if past is None and future is None:
            return super().forward(x)
        batch, _, height, width = x.shape
        rows, cols = (height + 31) // 32, (width + 31) // 32
        q = self._queries(x, rows, cols)
        k, v = self._tokens(self.key_value, x, rows, cols)
        out = self._attention(q, k, v)
        keys: list[Tensor] = []
        values: list[Tensor] = []
        for projection, state, valid in (
            (self.key_value_past, past, past_valid),
            (self.key_value_future, future, future_valid),
        ):
            if state is not None:
                k, v = self._tokens(
                    projection, state, rows, cols, valid, self.temporal_gain
                )
                keys.append(k)
                values.append(v)
        # The null token closes the temporal softmax and carries a zero value.
        null = torch.cat((self.null_key, self.null_bias[:, None]), dim=1)
        keys.append(null.to(k.dtype)[None, :, None].expand(q.shape[0], -1, -1, -1))
        values.append(v.new_zeros(q.shape[0], self.heads, 1, self.head_dim))
        out = out + self._attention(q, torch.cat(keys, dim=2), torch.cat(values, dim=2))
        return self.restore(out, batch, rows * 32, cols * 32)[..., :height, :width]


@store_hyperparameters()
class RSR(WESR):
    """WSR with a bidirectional recurrent read at the Halo stage.

    encode() runs a frame up to, but not including, the Halo stage, so it is
    shared by both sweeps. step_encoded() performs the frame's own and
    temporal Halo read, the remaining blocks and gates, and returns the state
    (the trunk after the last gate, the tensor that enters reconstruct) and
    optionally the image. forward() accepts one frame (B,3,H,W) or a clip
    (B,T,3,H,W); infer() gives the clip result for long videos in chunks.

    temporal_read "block" reads neighbours through block tokens in the Halo
    attention; "aligned" reads them through flow after the frame's own Halo
    read, with the guidance computed by guidance() when none is passed.
    """

    def __init__(
        self,
        *,
        depth: int = 7,
        gates: int = 3,
        width: int = 64,
        hidden: int = 128,
        context_width: int = 64,
        heads: int = 2,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        temporal_read: Literal["block", "aligned"] = "block",
    ) -> None:
        super().__init__(
            depth=depth,
            gates=gates,
            width=width,
            hidden=hidden,
            context_width=context_width,
            heads=heads,
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            scale=scale,
        )
        if self.merge_position > self.xsa_position or any(
            q < 2
            for p, q in zip(self.gate_positions, self.context_indices, strict=True)
            if p >= self.xsa_position
        ):
            raise ValueError(
                "RSR requires the merge before the Halo stage and merged context after it"
            )
        if temporal_read not in ("block", "aligned"):
            raise ValueError(
                f"temporal_read must be 'block' or 'aligned', not {temporal_read!r}"
            )
        self.temporal_read = temporal_read
        if temporal_read == "block":
            self.attention = TemporalBlockTokenAttention(channels=width, heads=heads)
            return
        # The reverse sweep reads the future, the forward sweep both neighbours.
        self.blend_reverse = nn.Conv2d(2 * width, width, 3, padding=1)
        self.blend_forward = nn.Conv2d(3 * width, width, 3, padding=1)
        for blend in (self.blend_reverse, self.blend_forward):
            nn.init.zeros_(blend.weight)
            nn.init.zeros_(blend.bias)
        # softplus(k) starts at 0.5 for both neighbours.
        start = math.log(math.expm1(0.5))
        self.confidence_past = nn.Parameter(torch.tensor(start))
        self.confidence_future = nn.Parameter(torch.tensor(start))
        # The frozen flow estimator comes last, on a forked RNG, so a seed still
        # gives every other initial weight; load_flow_weights() fills it.
        # RDUAL's module imports Triton and Flex attention: aligned read only.
        from ...RDUAL.__arch.rdual_arch import NeuFlowV2

        with torch.random.fork_rng(devices=[]):
            self.flow = NeuFlowV2()
        self.flow.requires_grad_(False)
        self.flow.eval()
        self.register_buffer("flow_loaded", torch.tensor(False))

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> RSR:
        # The frozen flow estimator follows the model's device but stays in
        # FP32 when the model is cast (half(), to(dtype)): guidance() runs FP32.
        if self.temporal_read != "aligned" or not recurse:
            return super()._apply(fn, recurse)
        flow = self._modules.pop("flow")
        try:
            super()._apply(fn, recurse)
        finally:
            self._modules["flow"] = flow
        device = fn(torch.empty(0, device=self.flow_loaded.device)).device
        flow._apply(lambda t: t.to(device))
        return self

    def train(self, mode: bool = True) -> RSR:
        super().train(mode)
        if self.temporal_read == "aligned":
            # The flow estimator stays in eval mode: its BatchNorm statistics are fixed.
            self.flow.eval()
        return self

    def load_flow_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Load an upstream NeuFlow v2 state dict (no prefix) strictly into flow."""
        self.flow.load_state_dict(state_dict, strict=True)
        self.flow_loaded.fill_(True)

    @torch.no_grad()
    def guidance(self, frames: Tensor, shot_ids: Tensor | None = None) -> Tensor:
        """The aligned read's guidance of a B,T,3,H,W clip in 0..1: B,T,2,3,h,w.

        Entry [:, t, 0] reads frame t-1 into frame t and [:, t, 1] frame t+1:
        the flow from frame t to that neighbour in x/y pixels of the h,w grid,
        then its forward-backward inconsistency in frame pixels. Entries
        without a neighbour (clip ends, a change of shot id) are zero, and no
        flow is computed across a shot change.

        The flow runs on frames reduced by flow_working_factor() to h,w =
        round(H/f), round(W/f) (bilinear with antialiasing), in 0..255,
        padded by replication to a multiple of 16 and at least 32 pixels a
        side and cropped back. FP32 without gradients, with autocast and TF32
        off inside and the caller's settings restored, on the model's device
        in calls of at most _flow_pairs_per_call() pairs; the result is on the
        frames' device.
        """
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("RSR guidance expects B,T,3,H,W frames")
        if not bool(self.flow_loaded):
            raise RuntimeError(
                "RSR has no NeuFlow v2 weights: call load_flow_weights() with the "
                "upstream state dict, or pass a guidance tensor instead"
            )
        batch, count, _, height, width = frames.shape
        factor = flow_working_factor(height, width)
        flow_h, flow_w = max(1, round(height / factor)), max(1, round(width / factor))
        result = torch.zeros((batch, count, 2, 3, flow_h, flow_w), device=frames.device)
        if shot_ids is None:
            linked = torch.ones(batch, count - 1, dtype=torch.bool)
        else:
            linked = (shot_ids[:, :-1] == shot_ids[:, 1:]).cpu()
        # Pair k joins frames k % (T-1) and the next one of sample k // (T-1).
        index = linked.flatten().nonzero().squeeze(1).to(frames.device)
        if index.numel() == 0:
            return result
        device = self.flow_loaded.device
        padded_h = max(32, (flow_h + 15) // 16 * 16)
        padded_w = max(32, (flow_w + 15) // 16 * 16)
        earlier = frames[:, :-1].flatten(0, 1)
        later = frames[:, 1:].flatten(0, 1)
        # RDUAL's module imports Triton and Flex attention: aligned read only.
        from ...RDUAL.__arch.rdual_arch import _flow_pairs_per_call

        pairs = _flow_pairs_per_call(padded_h, padded_w)
        ahead = result.new_zeros((batch * (count - 1), 3, flow_h, flow_w))
        behind = torch.zeros_like(ahead)
        # The per-backend switches only: the global getters raise once a
        # process has set one backend on its own.
        matmul, conv = torch.backends.cuda.matmul, torch.backends.cudnn.conv
        matmul_precision, conv_precision = matmul.fp32_precision, conv.fp32_precision
        with torch.autocast(device.type, enabled=False):
            matmul.fp32_precision = "ieee"
            conv.fp32_precision = "ieee"
            try:
                for first in range(0, index.numel(), pairs):
                    span = index[first : first + pairs]
                    images = torch.cat((earlier[span], later[span]))
                    images = images.to(device, torch.float32)
                    if factor > 1:
                        images = F.interpolate(
                            images,
                            size=(flow_h, flow_w),
                            mode="bilinear",
                            antialias=True,
                            align_corners=False,
                        ).clamp(0, 1)
                    images = images * 255.0
                    if (padded_h, padded_w) != (flow_h, flow_w):
                        padding = (0, padded_w - flow_w, 0, padded_h - flow_h)
                        images = F.pad(images, padding, mode="replicate")
                    # Flow on the grid of the first image, towards the second.
                    flow = self.flow(images, images.roll(span.numel(), 0))
                    flow = flow[:, :, :flow_h, :flow_w]
                    forward, backward = flow.split(span.numel())
                    scale_x, scale_y = width / flow_w, height / flow_h
                    ahead[span] = torch.cat(
                        (forward, _inconsistency(forward, backward, scale_x, scale_y)),
                        1,
                    ).to(frames.device)
                    behind[span] = torch.cat(
                        (backward, _inconsistency(backward, forward, scale_x, scale_y)),
                        1,
                    ).to(frames.device)
            finally:
                matmul.fp32_precision = matmul_precision
                conv.fp32_precision = conv_precision
        result[:, :-1, 1] = ahead.view(batch, count - 1, 3, flow_h, flow_w)
        result[:, 1:, 0] = behind.view(batch, count - 1, 3, flow_h, flow_w)
        return result

    def _clip_guidance(
        self, frames: Tensor, shot_ids: Tensor | None, guidance: Tensor | None
    ) -> Tensor | None:
        """The guidance a clip runs with: None for the block read."""
        if self.temporal_read == "block":
            if guidance is not None:
                raise ValueError("guidance is used with temporal_read='aligned' only")
            return None
        return self.guidance(frames, shot_ids) if guidance is None else guidance

    def encode(self, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run positions 1 to the Halo stage, excluding its attention and gate."""
        height, width = frame.shape[-2:]
        padding = (0, (-width) % 4, 0, (-height) % 4)
        current = self.stem(frame) + self.first_project(
            F.relu(self.first_expand(frame))
        )
        context = F.relu(self.context_stem(F.pixel_unshuffle(F.pad(frame, padding), 4)))
        c1 = self.context_blocks[0](context)
        c2 = self.context_blocks[1](c1)
        contexts = [c1, c2]
        for position in range(1, self.xsa_position + 1):
            if position > 1:
                current = self.blocks[position - 2](current)
            if position == self.merge_position:
                c3 = self.context_blocks[2](c2 + self.merge(F.pad(current, padding)))
                c4 = self.context_blocks[3](c3)
                contexts.extend((c3, c4))
            if position in self.gate_at and position < self.xsa_position:
                j, q = self.gate_at[position]
                current = self.gates[j](current, contexts[q])
        return current, contexts[2], contexts[3]

    def step_encoded(
        self,
        frame: Tensor,
        encoded: tuple[Tensor, Tensor, Tensor],
        past: Tensor | None = None,
        future: Tensor | None = None,
        past_valid: Tensor | None = None,
        future_valid: Tensor | None = None,
        past_guidance: Tensor | None = None,
        future_guidance: Tensor | None = None,
        *,
        output: bool = True,
        reverse: bool = False,
    ) -> tuple[Tensor | None, Tensor]:
        """Finish one encoded frame; return (image or None, state).

        The guidance (B,3,h,w per neighbour) and reverse (a reverse-sweep step,
        which reads the future only) are used by the aligned read.
        """
        state = self._state(
            encoded,
            past,
            future,
            past_valid,
            future_valid,
            past_guidance,
            future_guidance,
            reverse=reverse,
        )
        if not output:
            return None, state
        return self.head(frame, state), state

    def _state(
        self,
        encoded: Encoded,
        past: Tensor | None,
        future: Tensor | None,
        past_valid: Tensor | None,
        future_valid: Tensor | None,
        past_guidance: Tensor | None,
        future_guidance: Tensor | None,
        *,
        reverse: bool = False,
    ) -> Tensor:
        """Own and temporal Halo read, then the remaining blocks and gates."""
        current, c3, c4 = encoded
        if self.temporal_read == "block":
            current = current + self.attention(
                current, past, future, past_valid, future_valid
            )
        else:
            current = current + self.attention(current)
            reads = [
                self.aligned_read(
                    current,
                    future,
                    future_valid,
                    future_guidance,
                    self.confidence_future,
                )
            ]
            if not reverse:
                reads.insert(
                    0,
                    self.aligned_read(
                        current, past, past_valid, past_guidance, self.confidence_past
                    ),
                )
            blend = self.blend_reverse if reverse else self.blend_forward
            current = current + blend(torch.cat((current, *reads), dim=1))
        for position in range(self.xsa_position, self.depth + 1):
            if position > self.xsa_position:
                current = self.blocks[position - 2](current)
            if position in self.gate_at:
                j, q = self.gate_at[position]
                current = self.gates[j](current, (c3, c4)[q - 2])
        return current

    @staticmethod
    def aligned_read(
        current: Tensor,
        state: Tensor | None,
        valid: Tensor | None,
        guidance: Tensor | None,
        confidence: Tensor,
    ) -> Tensor:
        """A neighbour's tanh(state) warped onto the frame, times its confidence.

        guidance (B,3,h,w) is resized to the state's size (bilinear,
        align_corners=False, flow times the size ratio); the state is sampled
        bilinearly at p + flow with zeros outside the frame and weighted by
        exp(-softplus(confidence) * inconsistency^2). Grid and sampling run in
        FP32 (FP64 for double features) outside autocast; the read returns in
        the features' dtype. An absent neighbour (None, or False in valid for
        a sample) reads zero.
        """
        if state is None:
            return torch.zeros_like(current)
        if guidance is None:
            raise ValueError("the aligned read of a neighbour needs its guidance")
        height, width = current.shape[-2:]
        precision = torch.promote_types(current.dtype, torch.float32)
        bounded = torch.tanh(state)
        with torch.autocast(device_type=current.device.type, enabled=False):
            guide = guidance.to(precision)
            flow_h, flow_w = guide.shape[-2:]
            guide = F.interpolate(
                guide, size=(height, width), mode="bilinear", align_corners=False
            )
            y, x = torch.meshgrid(
                torch.arange(height, dtype=precision, device=guide.device),
                torch.arange(width, dtype=precision, device=guide.device),
                indexing="ij",
            )
            x = x + guide[:, 0] * (width / flow_w)
            y = y + guide[:, 1] * (height / flow_h)
            # align_corners=False: pixel i sits at (2i + 1) / size - 1.
            grid = torch.stack(
                ((2 * x + 1) / width - 1, (2 * y + 1) / height - 1), dim=-1
            )
            warped = F.grid_sample(
                bounded.to(precision),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            weight = torch.exp(
                -F.softplus(confidence.to(precision)) * guide[:, 2:].square()
            )
            read = weight * warped
        if valid is not None:
            read = torch.where(valid.view(-1, 1, 1, 1), read, 0.0)
        return read.to(current.dtype)

    def head(self, frame: Tensor, state: Tensor) -> Tensor:
        """Reconstruct the image of a frame from its state."""
        return F.pixel_shuffle(self.reconstruct(state) + self.base(frame), self.scale)

    @staticmethod
    def _links(shot_ids: Tensor | None, start: int, stop: int) -> list[Tensor | None]:
        """Per-sample validity between frames t and t+1 for t in [start, stop)."""
        if shot_ids is None:
            return [None] * (stop - start)
        links: list[Tensor | None] = [
            shot_ids[:, t] == shot_ids[:, t + 1] for t in range(start, stop - 1)
        ]
        return [*links, None]

    def _sweep(
        self,
        frames: list[Tensor],
        encoded: list[tuple[Tensor, Tensor, Tensor]],
        links: list[Tensor | None],
        count: int,
        past: Tensor | None = None,
        past_valid: Tensor | None = None,
        guidance: Tensor | None = None,
    ) -> tuple[list[Tensor], Tensor | None]:
        """Reverse sweep over the whole window, forward sweep over its first frames.

        links[t] is the validity between window frames t and t+1; past is the
        forward state of the frame before the window, valid per past_valid.
        guidance (B,L,2,3,h,w), the window's, is given for the aligned read.
        """

        def guide(t: int, side: int) -> Tensor | None:
            return None if guidance is None else guidance[:, t, side]

        length = len(frames)
        reverse: list[Tensor | None] = [None] * (length + 1)
        for t in range(length - 1, 0, -1):
            _, reverse[t] = self.step_encoded(
                frames[t],
                encoded[t],
                future=reverse[t + 1],
                future_valid=links[t],
                future_guidance=guide(t, 1),
                output=False,
                reverse=True,
            )
        outputs: list[Tensor] = []
        state = past
        for t in range(count):
            _, state = self.step_encoded(
                frames[t],
                encoded[t],
                state,
                reverse[t + 1],
                past_valid if t == 0 else links[t - 1],
                links[t],
                guide(t, 0),
                guide(t, 1),
                output=False,
            )
            outputs.append(self.head(frames[t], state))
        return outputs, state

    def forward(
        self,
        x: Tensor,
        shot_ids: Tensor | None = None,
        guidance: Tensor | None = None,
    ) -> Tensor:
        """One frame (B,3,H,W) without neighbours, or a clip (B,T,3,H,W).

        shot_ids (B,T), optional: no state crosses a change of shot id.
        guidance (B,T,2,3,h,w), aligned read only: computed by guidance()
        when None.
        """
        if x.ndim == 4:
            _, state = self.step_encoded(x, self.encode(x), output=False)
            return self.head(x, state)
        if x.ndim != 5:
            raise ValueError("RSR expects (B,3,H,W) or (B,T,3,H,W) input")
        frames = list(x.unbind(1))
        outputs, _ = self._sweep(
            frames,
            [self.encode(frame) for frame in frames],
            self._links(shot_ids, 0, len(frames)),
            len(frames),
            guidance=self._clip_guidance(x, shot_ids, guidance),
        )
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def infer(
        self,
        frames: Tensor,
        chunk: int = 16,
        lookahead: int = 8,
        shot_ids: Tensor | None = None,
        guidance: Tensor | None = None,
    ) -> Tensor:
        """Clip result in bounded memory: chunks with a look-ahead reverse sweep.

        The forward state is carried across chunks; each chunk's reverse sweep
        starts lookahead frames past its end. Every frame is encoded once.
        Equals forward() exactly when chunk + lookahead covers the clip.
        For the aligned read, guidance None computes each chunk's guidance
        from its frames, the look-ahead and the frame before the chunk, so
        the flow across a chunk edge exists; a tensor is sliced per chunk.
        """
        if frames.ndim != 5 or chunk < 1 or lookahead < 0:
            raise ValueError("infer expects (B,T,3,H,W), chunk >= 1, lookahead >= 0")
        count = frames.shape[1]
        encoded: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        outputs: list[Tensor] = []
        past: Tensor | None = None
        for start in range(0, count, chunk):
            end = min(start + chunk, count)
            stop = min(end + lookahead, count)
            for t in range(start, stop):
                if t not in encoded:
                    encoded[t] = self.encode(frames[:, t])
            past_valid = (
                None
                if shot_ids is None or start == 0
                else shot_ids[:, start - 1] == shot_ids[:, start]
            )
            # The frame before the chunk links the carried state through flow.
            first = max(start - 1, 0)
            window = self._clip_guidance(
                frames[:, first:stop],
                None if shot_ids is None else shot_ids[:, first:stop],
                None if guidance is None else guidance[:, first:stop],
            )
            chunk_outputs, past = self._sweep(
                list(frames[:, start:stop].unbind(1)),
                [encoded[t] for t in range(start, stop)],
                self._links(shot_ids, start, stop),
                end - start,
                past,
                past_valid,
                None if window is None else window[:, start - first :],
            )
            outputs.extend(chunk_outputs)
            for t in range(start, end):
                encoded.pop(t)
        return torch.stack(outputs, dim=1)

    def step_functions(
        self, compile: Callable[[Callable], Callable] | None = None
    ) -> StepFunctions:
        """The callables backward_clip() runs, each wrapped by compile when given.

        compile is for example torch.compile or a functools.partial of it with a
        mode; None keeps every callable eager. They do not depend on the clip
        length, so one set serves every clip. Each takes the guidance of the
        neighbours it reads last (sg past, dg future; None for the block read).
        """

        def full(
            x: Tensor,
            e: Encoded,
            s: Tensor,
            v: Tensor,
            d: Tensor,
            lv: Tensor,
            sg: Tensor | None,
            dg: Tensor | None,
        ) -> tuple[Tensor, Tensor]:
            state = self._state(e, s, d, v, lv, sg, dg)
            return self.head(x, state), state

        def reverse(e: Encoded, d: Tensor, lv: Tensor, dg: Tensor | None) -> Tensor:
            return self._state(e, None, d, None, lv, None, dg, reverse=True)

        def forward_state(
            e: Encoded,
            s: Tensor,
            v: Tensor,
            d: Tensor,
            lv: Tensor,
            sg: Tensor | None,
            dg: Tensor | None,
        ) -> Tensor:
            return self._state(e, s, d, v, lv, sg, dg)

        if compile is None:
            return StepFunctions(self.encode, full, reverse, forward_state)
        return StepFunctions(
            compile(self.encode),
            compile(full),
            compile(reverse),
            compile(forward_state),
        )

    def backward_clip(
        self,
        frames: Tensor,
        loss: Callable[[Tensor, int], Tensor],
        *,
        shot_ids: Tensor | None = None,
        guidance: Tensor | None = None,
        keep: str = "none",
        functions: StepFunctions | None = None,
        autocast: Callable[[], AbstractContextManager] = nullcontext,
        grad_chunk: int = 2,
        encode_chunk: int = 5,
    ) -> Tensor:
        """Run one clip and backpropagate the sum of its per-frame losses.

        Parameter and frame gradients equal those of summing loss(out[:, t], t)
        over self(frames, shot_ids, guidance) and calling backward, and accumulate into
        .grad. loss is called once per frame, inside autocast(), with that
        frame's B,3,sH,sW output as a detached leaf, so it must not rely on
        reaching the model through it. A frame's loss graph is released as soon
        as its gradient at the output is known, so held modes hold only the
        model's graphs. Returns the detached outputs, B,T,3,sH,sW.

        An absent neighbour (before the first frame, after the last, across a
        shot change) is a zero state behind a closed per-sample mask, so every
        frame of a clip has the same call shape; forward() omits those tokens
        instead, which differs only by the summation order of the softmax.

        keep chooses what is held instead of recomputed: "none" (no-gradient
        sweeps, then one replay per frame and direction), "reverse" (the reverse
        sweep's graphs are held until their backward) or "both" (every graph is
        held; no no-gradient pass and no replay). Held graphs trade memory for
        speed.

        Frames are encoded once without gradients in calls of encode_chunk
        frames, shared by both sweeps, and with gradients at the end in calls of
        grad_chunk frames; a clip length either does not divide gives one more
        call shape, which a compiled encode compiles separately. functions
        comes from step_functions() (eager when None).

        For CUDA frames the method marks its own CUDA-graph steps
        (torch.compiler.cudagraph_mark_step_begin(), a no-op without CUDA
        graphs): each no-gradient encode, reverse and forward-state call, each
        frame of the forward and reverse replays (forward, loss and both
        backwards together) and each gradient encode chunk. Everything that
        outlives a step is copied into ordinary memory, and every parameter that
        requires gradients gets a zero .grad before the clip, so one that
        receives nothing ends the clip with zeros instead of None. With
        keep="none" each gradient leaf (a frame's encoded features, a state, a
        donor) also gets a zero gradient just before the step that first writes
        it, which later steps add to in place; held modes leave leaf gradients
        to autograd inside their span and copy the encoded features' gradients
        out of it before the gradient encode. Under CUDA graphs keep="none" is
        the supported mode: held modes mark only the steps outside the span
        their held graphs cover, which then runs as one step. A loss callable
        may run compiled functions, which belong to the frame's step, but must
        not keep their outputs beyond the call without copying them; the caller
        marks steps for its own compiled calls outside this method.

        For the aligned read, guidance None computes it with guidance() first,
        in eager code before any step mark; a tensor is used as given.
        """
        if keep not in ("none", "reverse", "both"):
            raise ValueError(f"keep must be none/reverse/both, not {keep!r}")
        if grad_chunk < 1 or encode_chunk < 1:
            raise ValueError("grad_chunk and encode_chunk must be positive")
        if frames.ndim != 5:
            raise ValueError("backward_clip expects (B,T,3,H,W) frames")
        fn = self.step_functions() if functions is None else functions
        guidance = self._clip_guidance(frames, shot_ids, guidance)

        batch, count = frames.shape[:2]
        # A compiled step guards on strides, so every per-frame slice is contiguous.
        images = [frames[:, t].contiguous() for t in range(count)]
        past_guidance: list[Tensor | None] = [None] * count
        future_guidance: list[Tensor | None] = [None] * count
        if guidance is not None:
            past_guidance = [guidance[:, t, 0].contiguous() for t in range(count)]
            future_guidance = [guidance[:, t, 1].contiguous() for t in range(count)]
        closed = torch.zeros(batch, dtype=torch.bool, device=frames.device)
        if shot_ids is None:
            edge = [torch.ones_like(closed) for _ in range(count - 1)]
        else:
            same = (shot_ids[:, :-1] == shot_ids[:, 1:]).to(frames.device)
            edge = [same[:, t].contiguous() for t in range(count - 1)]
        past_valid = [closed, *edge]
        ahead_valid = [*edge, closed]
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
            # encode chunks add to parameter gradients in steps of their own,
            # after the held span's mark.
            for parameter in self.parameters():
                if parameter.requires_grad and parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)

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
        # frame: its forward step and its reverse step. Their gradients collect
        # both adjoints, and one encode with gradients at the end pushes the sum
        # into the parameters and frames, exact by linearity of the backward.
        # Each frame's features get their gradient buffers only when its first
        # gradient step comes.
        encoded = [tuple(x.requires_grad_() for x in features) for features in encoded]
        zero = torch.zeros_like(encoded[0][0])
        future = [zero] * (count + 1)
        states = [zero]

        def frame(
            t: int, state: Tensor, donor: Tensor
        ) -> tuple[Tensor, Tensor, Tensor]:
            """The output, the loss gradient there, the carried state.

            The loss backpropagates at once into a detached copy of the output,
            so its graph is gone before the model's is used.
            """
            with autocast():
                output, carried = fn.full(
                    images[t],
                    encoded[t],
                    state,
                    past_valid[t],
                    donor,
                    ahead_valid[t],
                    past_guidance[t],
                    future_guidance[t],
                )
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
                        encoded[t], leaves[t], ahead_valid[t], future_guidance[t]
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
                        ahead_valid[t],
                        future_guidance[t],
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
                            past_valid[t],
                            future[t + 1],
                            ahead_valid[t],
                            past_guidance[t],
                            future_guidance[t],
                        ).clone()
                    )
        # Backward runs from the last frame so each carried state's adjoint is
        # known before its producer's graph is built or released.
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
        # The reverse chain runs from frame 1 forwards: frame t's reverse state
        # feeds frame t-1's steps, so its adjoint is what the forward replays
        # owed it plus what frame t-1's reverse step passed back.
        carry: Tensor | None = None
        for t in range(1, count):
            if keep == "none":
                step()
                donor = leaf(future[t + 1])
                with autocast():
                    out = fn.reverse(
                        encoded[t], donor, ahead_valid[t], future_guidance[t]
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
        # grad_chunk frames per gradient encode call, each chunk's features and
        # adjoints released before the next.
        for first in range(0, count, grad_chunk):
            step()
            span = encoded[first : first + grad_chunk]
            with autocast():
                together = fn.encode(torch.cat(images[first : first + grad_chunk]))
            grads = [
                torch.cat([_gradient(features[k]) for features in span])
                for k in range(len(together))
            ]
            encoded[first : first + grad_chunk] = [()] * len(span)
            del span
            # The frames have no graph of their own unless they need gradients.
            roots = [
                (x, g) for x, g in zip(together, grads, strict=True) if x.requires_grad
            ]
            torch.autograd.backward([x for x, _ in roots], [g for _, g in roots])
            del together, grads, roots
        return torch.stack([outputs[t] for t in range(count)], 1)


# Registered preset.


def rsr_compact(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    scale: int = 4,
    temporal_read: Literal["block", "aligned"] = "block",
) -> RSR:
    """Recurrent WSR Compact C32/H64/D9/S3 with temporal reads; 1x/2x/4x."""
    return RSR(
        width=32,
        hidden=64,
        depth=9,
        gates=3,
        context_width=64,
        heads=1,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
        temporal_read=temporal_read,
    )
