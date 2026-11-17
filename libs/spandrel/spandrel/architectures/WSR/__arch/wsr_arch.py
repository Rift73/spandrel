"""WSR: wide residual super-resolution with multiscale context and Halo attention.

All five RGB presets support native 1x, 2x and 4x output. For input (B,3,H,W),
the output is (B,3,scale*H,scale*W); arbitrary positive H/W are padded internally
where needed and restored without resizing the image to a window multiple.

Preset              Width  Hidden  Depth  Gates  Heads  Context width
Default               64     128      7      3      2        64
Compact               32      64      9      3      1        64
UltraCompact          32      64      5      2      -        64
SuperUltraCompact     16      32      5      2      -        32
Mobile                 8      16      2      1      -        32

Default/Compact combine a full-resolution trunk, an LR/4 context stream and
Halo retrieval. The three smaller presets retain the trunk/context pair.
A pixel-shuffle head sets the output scale; trained head weights are
scale-specific even though the backbone design supports all three scales.

Keep training checkpoints unfolded. For deployment, make a separate copy,
call eval(), then prepare_for_export() to compose linear branches and cache
attention temperature. This conversion is in-place and is not reversible.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

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


# Attention-enabled and context-only backbones.


@store_hyperparameters()
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

    hyperparameters = {}

    def __init__(
        self,
        *,
        depth: int = 7,
        gates: int = 3,
        width: int = 64,
        hidden: int = 128,
        context_width: int = 64,
        heads: int = 2,
        use_xsa: bool = True,
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
        if use_xsa and not eligible:
            raise ValueError(
                "Attention needs a gate at least two blocks before the tail"
            )
        # use_xsa=False: the twin without attention.
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
        self.attention = (
            BlockTokenAttention(channels=width, heads=heads) if use_xsa else None
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


@store_hyperparameters()
class WideContextSR(nn.Module):
    """Context-only backbone for UltraCompact, SuperUltraCompact and Mobile.

    An HxW residual trunk exchanges features with an LR/4 context stream.
    Four context blocks supply scheduled channel-wise gates; the later two
    consume the midpoint trunk merge. A shallow residual and pixel-shuffle head
    produce (B,3,scale*H,scale*W), including unchanged resolution at scale=1.
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
            # Merge precedes later context gates, even when no gate occurs here.
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

        _fold_modules(self)
        self.deploy = True
        return self


# Registered presets.


def wsr(num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4) -> WESR:
    """Default C64/H128/D7/S3 with Halo attention; native 1x/2x/4x."""
    return WESR(num_in_ch=num_in_ch, num_out_ch=num_out_ch, scale=scale)


def wsr_compact(num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4) -> WESR:
    """Compact C32/H64/D9/S3 with Halo attention; native 1x/2x/4x."""
    return WESR(
        width=32,
        hidden=64,
        depth=9,
        gates=3,
        context_width=64,
        heads=1,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        scale=scale,
    )


def wsr_ultracompact(
    num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4
) -> WideContextSR:
    """UltraCompact: context-only WSR C32/H64/D5/S2."""
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
    """SuperUltraCompact: context-only WSR C16/H32/D5/S2."""
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
    """Mobile: context-only C8/H16/D2 with one gate and context width 32."""
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
