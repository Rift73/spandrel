from __future__ import annotations

import math

import torch
import torch.nn as nn
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.drft_arch import DRFT, ChannelAttention, EDBBConvBlock


class _FP16SafeChannelAttention(ChannelAttention):
    """Keep the tiny pooled SwiGLU path finite without upcasting image features."""

    def __init__(self, source: ChannelAttention) -> None:
        nn.Module.__init__(self)
        # Reuse loaded parameters under exactly the same state-dict keys.
        self.pool = source.pool
        self.project = source.project
        self.expand = source.expand
        self.sigmoid = source.sigmoid
        self.force_tensorrt_export_mode = source.force_tensorrt_export_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or x.dtype != torch.float16:
            return super().forward(x)

        # Trained DRFT can overflow SiLU(gate)*value in FP16 even though the
        # eventual sigmoid is bounded. Preserve the values in FP32 through
        # BOTH projections and the product; clamping would change the model.
        # Only B*C*1*1 features and these small 1x1 weights are promoted.
        # Persistent parameters and the full-resolution image remain FP16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            pooled = (
                x.mean(dim=(2, 3), keepdim=True)
                if self.force_tensorrt_export_mode
                else self.pool(x)
            ).float()
            gate, value = torch.conv2d(
                pooled,
                self.project.weight.float(),
                self.project.bias.float(),
            ).chunk(2, dim=1)
            y = nn.functional.silu(gate) * value
            y = torch.conv2d(y, self.expand.weight.float(), self.expand.bias.float())
            attention = self.sigmoid(y).to(dtype=x.dtype)
        return x * attention


def _get_progressive_upscale(state_dict: StateDict, num_feat: int) -> int:
    """Detect the scale of the progressive PixelAttention reconstruction head."""
    if "upsample.up.6.weight" in state_dict:
        return 8
    if "upsample.up.3.weight" in state_dict:
        return 4
    if "upsample.up.0.weight" in state_dict:
        out_channels = state_dict["upsample.up.0.weight"].shape[0]  # type: ignore
        return 3 if out_channels == 9 * num_feat else 2
    return 1


def _get_drft_upscale(
    state_dict: StateDict,
    *,
    in_chans: int,
    num_feat: int,
) -> int:
    """Detect both progressive and direct PixelShuffle reconstruction heads."""
    if "conv_last.weight" in state_dict:
        return _get_progressive_upscale(state_dict, num_feat)

    direct_weight = state_dict.get("upsample.0.weight")
    if direct_weight is None:
        return 1

    out_channels = direct_weight.shape[0]  # type: ignore
    if out_channels % in_chans != 0:
        raise ValueError("Invalid DRFT direct reconstruction weight shape")
    scale_squared = out_channels // in_chans
    scale = math.isqrt(scale_squared)
    if scale * scale != scale_squared:
        raise ValueError("Invalid DRFT direct PixelShuffle scale")
    return scale


def _detect_mlp_ratio(embed_dim: int, hidden_dim: int) -> float:
    """Find the MLP ratio that reproduces the rounded SwiGLU width."""
    nice = [2.667, 2.0, 4.0, 3.0, 1.5, 2.5, 3.5]
    for ratio in nice:
        if ((int(embed_dim * ratio) + 7) // 8) * 8 == hidden_dim:
            return ratio
    return hidden_dim / embed_dim


def _detect_residual_connection(state_dict: StateDict) -> str:
    if "layers.0.conv.weight" in state_dict:
        return "1conv"
    if "layers.0.conv.0.weight" in state_dict:
        return "3conv"
    return "identity"


def _is_legacy_checkpoint(state_dict: StateDict) -> bool:
    """Identify the pre-i-LN-family DRFT parameter contract."""
    block = "layers.0.residual_group.blocks.0.conv_block.0"
    return any(
        key in state_dict
        for key in (
            "_use_iln",
            "_use_edbb",
            f"{block}.branch_weight",
            f"{block}.identity_weight",
        )
    )


def _get_structure(state_dict: StateDict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    num_layers = get_seq_len(state_dict, "layers")
    depths = tuple(
        get_seq_len(state_dict, f"layers.{index}.residual_group.blocks")
        for index in range(num_layers)
    )
    num_heads = tuple(
        state_dict[
            f"layers.{index}.residual_group.ocab.relative_position_bias_table"
        ].shape[1]  # type: ignore
        for index in range(num_layers)
    )
    return depths, num_heads


def _get_window_geometry(state_dict: StateDict) -> tuple[int, int]:
    q_coords = state_dict[
        "layers.0.residual_group.blocks.0.attn.neural_bias.q_coords_table"
    ]
    window_size = math.isqrt(q_coords.shape[0])  # type: ignore

    relative_bias = state_dict[
        "layers.0.residual_group.ocab.relative_position_bias_table"
    ]
    bias_span = math.isqrt(relative_bias.shape[0])  # type: ignore
    overlap_window_size = bias_span - window_size + 1
    return window_size, overlap_window_size


def _get_current_window_geometry(
    state_dict: StateDict,
    depths: tuple[int, ...],
) -> tuple[int, tuple[int, ...], int]:
    """Recover ACT windows separately from the canonical OCAB geometry.

    Cyclic DRFT changes only ACT windows; OCAB keeps its half-window halo.
    Its table has (q + k - 1)**2 rows, with k = 3*q/2. Uniform checkpoints
    retain the historical inference rule, including non-default OCAB halos.
    Unidentifiable custom cyclic halos must not silently load as another model.
    """
    window_sizes: list[int] = []
    bias_spans: set[int] = set()
    for stage, depth in enumerate(depths):
        stage_windows: set[int] = set()
        for block in range(depth):
            prefix = f"layers.{stage}.residual_group.blocks.{block}." "attn.neural_bias"
            for suffix in ("q_coords_table", "k_coords_table"):
                coords = state_dict[f"{prefix}.{suffix}"]
                tokens = coords.shape[0]  # type: ignore
                window = math.isqrt(tokens)
                if window <= 0 or window * window != tokens or coords.shape[1:] != (2,):  # type: ignore
                    raise ValueError(f"Invalid DRFT coordinate grid: {prefix}.{suffix}")
                stage_windows.add(window)
        if len(stage_windows) != 1:
            raise ValueError(f"Inconsistent DRFT ACT windows in group {stage}")
        window_sizes.append(stage_windows.pop())

        table = state_dict[
            f"layers.{stage}.residual_group.ocab.relative_position_bias_table"
        ]
        rows = table.shape[0]  # type: ignore
        span = math.isqrt(rows)
        if span * span != rows:
            raise ValueError(f"Invalid DRFT OCAB relative-bias table in group {stage}")
        bias_spans.add(span)

    if len(bias_spans) != 1:
        raise ValueError("DRFT requires the same OCAB geometry in every group")
    bias_span = bias_spans.pop()
    if len(set(window_sizes)) == 1:
        window_size = window_sizes[0]
    else:
        window_size, remainder = divmod(2 * (bias_span + 1), 5)
        if remainder or window_size not in window_sizes:
            raise ValueError(
                "Cannot infer nonstandard cyclic DRFT OCAB geometry from this "
                "checkpoint; canonical cyclic checkpoints use a half-window halo"
            )
    overlap_window_size = bias_span - window_size + 1
    if (
        window_size <= 0
        or any(window % 2 for window in window_sizes)
        or window_size % 2
        or overlap_window_size < window_size
        or (overlap_window_size - window_size) % 2
    ):
        raise ValueError("Invalid DRFT ACT/OCAB window geometry")
    return window_size, tuple(window_sizes), overlap_window_size


def _size_tag(num_layers: int) -> str:
    if num_layers >= 10:
        return "large"
    if num_layers >= 6:
        return "medium"
    if num_layers >= 4:
        return "small"
    return "extra-small"


def _descriptor(
    architecture: DRFTArch,
    model: nn.Module,
    state_dict: StateDict,
    *,
    upscale: int,
    in_chans: int,
    tags: list[str],
) -> ImageModelDescriptor[nn.Module]:
    return ImageModelDescriptor(
        model,
        state_dict,
        architecture=architecture,
        purpose="Restoration" if upscale == 1 else "SR",
        tags=tags,
        supports_half=True,
        supports_bfloat16=True,
        scale=upscale,
        input_channels=in_chans,
        output_channels=in_chans,
        size_requirements=SizeRequirements(minimum=16),
    )


def _detect_channel_squeeze_factor(state_dict: StateDict, embed_dim: int) -> int:
    project = state_dict["layers.0.residual_group.blocks.0.conv_block.3.project.weight"]
    projected_channels = project.shape[0]  # type: ignore
    if projected_channels % 2 != 0:
        raise ValueError("Invalid DRFT ChannelAttention projection width")
    hidden = projected_channels // 2

    preferred = (16, 8, 4, 32, 2, 1)
    for factor in (*preferred, *range(1, embed_dim + 1)):
        if max(1, embed_dim // factor) == hidden:
            return factor
    raise ValueError("Unable to infer DRFT ChannelAttention squeeze factor")


def _get_current_attention_layout(
    state_dict: StateDict,
    *,
    embed_dim: int,
    depths: tuple[int, ...],
    num_heads: tuple[int, ...],
) -> tuple[int, tuple[int, ...], tuple[int, ...], bool]:
    """Infer rank and the optional narrow-unshifted layout from tensor shapes."""
    rank: int | None = None
    for stage, depth in enumerate(depths):
        if depth < 2:
            continue
        bias = state_dict[
            f"layers.{stage}.residual_group.blocks.1."
            "attn.neural_bias.bias_mlp_qk.2.weight"
        ]
        divisor = num_heads[stage] * 2
        if bias.shape[0] % divisor == 0:  # type: ignore
            rank = bias.shape[0] // divisor  # type: ignore
            break
    if rank is None:
        bias = state_dict[
            "layers.0.residual_group.blocks.0." "attn.neural_bias.bias_mlp_qk.2.weight"
        ]
        divisor = num_heads[0] * 2
        if bias.shape[0] % divisor != 0:  # type: ignore
            raise ValueError("Unable to infer DRFT neural-bias rank")
        rank = bias.shape[0] // divisor  # type: ignore

    unshifted_num_heads: list[int] = []
    unshifted_attention_dim: list[int] = []
    for stage in range(len(depths)):
        prefix = f"layers.{stage}.residual_group.blocks.0.attn"
        qkv = state_dict[f"{prefix}.qkv.weight"]
        bias = state_dict[f"{prefix}.neural_bias.bias_mlp_qk.2.weight"]
        attention_dim = qkv.shape[0] // 3  # type: ignore
        divisor = rank * 2
        if bias.shape[0] % divisor != 0:  # type: ignore
            raise ValueError("Invalid DRFT unshifted neural-bias shape")
        heads = bias.shape[0] // divisor  # type: ignore
        unshifted_attention_dim.append(attention_dim)
        unshifted_num_heads.append(heads)

    full_width = all(
        attention_dim == embed_dim and heads == num_heads[stage]
        for stage, (attention_dim, heads) in enumerate(
            zip(unshifted_attention_dim, unshifted_num_heads, strict=True)
        )
    )
    return (
        rank,
        tuple(unshifted_num_heads),
        tuple(unshifted_attention_dim),
        full_width,
    )


def _prepare_folded_edbb(model: DRFT) -> None:
    """Match a deployment-folded checkpoint without folding random weights."""
    blocks = [module for module in model.modules() if isinstance(module, EDBBConvBlock)]
    for block in blocks:
        reference = block.conv3x3.weight
        block._folded_conv = nn.Conv2d(
            block.dim,
            block.dim,
            3,
            1,
            1,
            bias=True,
            device=reference.device,
            dtype=reference.dtype,
        )
        block._clear_eval_cache()
        del block.conv3x3
        del block.conv1x1
        del block.conv1x1_3x3
        del block.conv1x1_sbx
        del block.conv1x1_sby
        del block.conv1x1_lpl
        block._is_folded = True


class DRFTArch(Architecture[nn.Module]):
    def __init__(self) -> None:
        super().__init__(
            id="DRFT",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "conv_after_body.weight",
                "layers.0.residual_group.blocks.0.attn.neural_bias.bias_mlp_qk.0.weight",
                "layers.0.residual_group.blocks.0.ffn.fc_gate_value.weight",
                "layers.0.residual_group.blocks.0.ls_attn.gamma",
                "layers.0.residual_group.ocab.relative_position_bias_table",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[nn.Module]:
        if _is_legacy_checkpoint(state_dict):
            return self._load_legacy(state_dict)
        return self._load_current(state_dict)

    def _load_legacy(self, state_dict: StateDict) -> ImageModelDescriptor[nn.Module]:
        # Import lazily so current DRFT models do not pay for the legacy module.
        # (a dotted module name, which Python does not mangle inside a class)
        from .__arch.legacy_drft_arch import DRFT as LEGACY_DRFT

        in_chans = state_dict["conv_first.weight"].shape[1]  # type: ignore
        embed_dim = state_dict["conv_first.weight"].shape[0]  # type: ignore
        num_feat = state_dict["conv_last.weight"].shape[1]  # type: ignore
        upscale = _get_drft_upscale(
            state_dict,
            in_chans=in_chans,
            num_feat=num_feat,
        )
        depths, num_heads = _get_structure(state_dict)
        window_size, overlap_window_size = _get_window_geometry(state_dict)
        overlap_ratio = (overlap_window_size - window_size) / window_size

        hidden_dim = (
            state_dict[
                "layers.0.residual_group.blocks.0.ffn.fc_gate_value.weight"
            ].shape[0]  # type: ignore
            // 2
        )
        mlp_ratio = _detect_mlp_ratio(embed_dim, hidden_dim)
        qkv_bias = "layers.0.residual_group.blocks.0.attn.qkv.bias" in state_dict
        bias_out = state_dict[
            "layers.0.residual_group.blocks.0." "attn.neural_bias.bias_mlp_qk.2.weight"
        ].shape[0]  # type: ignore
        rank = bias_out // (num_heads[0] * 2)
        dense_skip = "layers.0.residual_group.dense_fusion.weight" in state_dict
        resi_connection = _detect_residual_connection(state_dict)
        use_iln = bool(state_dict["_use_iln"]) if "_use_iln" in state_dict else False
        use_edbb = bool(state_dict["_use_edbb"]) if "_use_edbb" in state_dict else False
        folded = (
            "layers.0.residual_group.blocks.0."
            "conv_block.0._folded_conv.weight" in state_dict
        )

        model = LEGACY_DRFT(
            img_size=64,
            patch_size=1,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            conv_scale=0.01,
            layer_scale_init=1e-6,
            upscale=upscale,
            img_range=1.0,
            resi_connection=resi_connection,
            dense_skip=dense_skip,
            num_feat=num_feat,
            rank=rank,
            use_iln=use_iln,
            use_edbb=use_edbb,
            attn_type="masked",
        )

        tags = [
            _size_tag(len(depths)),
            "legacy",
            f"s64w{window_size}",
            f"{num_feat}nf",
            f"{embed_dim}dim",
            resi_connection,
        ]
        if use_iln:
            tags.append("iLN")
        if use_edbb:
            tags.append("EDBB")
        if folded:
            tags.append("folded")
        return _descriptor(
            self,
            model,
            state_dict,
            upscale=upscale,
            in_chans=in_chans,
            tags=tags,
        )

    def _load_current(self, state_dict: StateDict) -> ImageModelDescriptor[nn.Module]:
        in_chans = state_dict["conv_first.weight"].shape[1]  # type: ignore
        embed_dim = state_dict["conv_first.weight"].shape[0]  # type: ignore
        reconstruction = "progressive" if "conv_last.weight" in state_dict else "direct"
        num_feat = (
            state_dict["conv_last.weight"].shape[1]  # type: ignore
            if reconstruction == "progressive"
            else 64
        )
        upscale = _get_drft_upscale(
            state_dict,
            in_chans=in_chans,
            num_feat=num_feat,
        )
        depths, num_heads = _get_structure(state_dict)
        window_size, window_sizes, overlap_window_size = _get_current_window_geometry(
            state_dict,
            depths,
        )

        hidden_dim = (
            state_dict[
                "layers.0.residual_group.blocks.0.ffn.fc_gate_value.weight"
            ].shape[0]  # type: ignore
            // 2
        )
        mlp_ratio = _detect_mlp_ratio(embed_dim, hidden_dim)
        qkv_bias = "layers.0.residual_group.blocks.0.attn.qkv.bias" in state_dict
        rank, unshifted_heads, unshifted_dim, full_width = (
            _get_current_attention_layout(
                state_dict,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
            )
        )
        dense_skip = "layers.0.residual_group.dense_fusion.weight" in state_dict
        resi_connection = _detect_residual_connection(state_dict)
        folded = (
            "layers.0.residual_group.blocks.0."
            "conv_block.0._folded_conv.weight" in state_dict
        )

        edbb_depth_multiplier = 1.0
        edbb_probe = "layers.0.residual_group.blocks.0." "conv_block.0.conv1x1_3x3.k0"
        if edbb_probe in state_dict:
            edbb_depth_multiplier = state_dict[edbb_probe].shape[0] / embed_dim  # type: ignore

        channel_squeeze_factor = _detect_channel_squeeze_factor(
            state_dict,
            embed_dim,
        )
        rhag_layer_scale_init = 1e-4 if "layers.0.ls_rhag.gamma" in state_dict else None

        model = DRFT(
            img_size=64,
            patch_size=1,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            unshifted_num_heads=unshifted_heads,
            unshifted_attention_dim=unshifted_dim,
            window_size=window_size,
            window_sizes=window_sizes,
            overlap_window_size=overlap_window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            conv_scale=0.01,
            layer_scale_init=1e-6,
            rhag_layer_scale_init=rhag_layer_scale_init,
            use_checkpoint=False,
            upscale=upscale,
            img_range=1.0,
            resi_connection=resi_connection,
            dense_skip=dense_skip,
            num_feat=num_feat,
            rank=rank,
            attn_type="masked",
            reconstruction=reconstruction,
            full_width_unshifted=full_width,
            edbb_depth_multiplier=edbb_depth_multiplier,
            channel_squeeze_factor=channel_squeeze_factor,
            iln_eps=1e-4,
        )
        if folded:
            _prepare_folded_edbb(model)

        tags = [
            _size_tag(len(depths)),
            "iLN",
            "EDBB",
            "full-attn" if full_width else "narrow-attn",
            f"s64w{window_size}",
            f"{embed_dim}dim",
            resi_connection,
            reconstruction,
        ]
        if len(set(window_sizes)) > 1:
            tags.append("cyclic-windows")
        if reconstruction == "progressive":
            tags.append(f"{num_feat}nf")
        if folded:
            tags.append("folded")
        if rhag_layer_scale_init is not None:
            tags.append("RHAG-LS")

        descriptor = _descriptor(
            self,
            model,
            state_dict,
            upscale=upscale,
            in_chans=in_chans,
            tags=tags,
        )

        # Strict loading above must finish before branches are removed. This
        # is terminal preparation of the in-memory inference model only: no
        # checkpoint is saved and the caller's state dict is not rewritten.
        # Fold on CPU in FP32 before ModelLoader moves this module to the GPU
        # and chaiNNer optionally calls half(). Keep training's reversible
        # eval implementation in drft_arch.py unchanged.
        model.float().eval()
        with torch.no_grad():
            model.fold_reparam_conv()
        for name, module in list(model.named_modules()):
            if isinstance(module, EDBBConvBlock):
                # Permanent folding already incorporated the identity into
                # the 3x3 kernel; retaining this C*C*9 buffer wastes VRAM.
                del module.identity_weight
            elif isinstance(module, ChannelAttention):
                model.set_submodule(name, _FP16SafeChannelAttention(module))
        model.requires_grad_(False).eval()
        return descriptor


__all__ = ["DRFTArch", "DRFT"]
