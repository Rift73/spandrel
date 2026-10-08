from __future__ import annotations

import math

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.sst_arch import SST, EDMUNetSR

# =============================================================================
# SST — windowed super-resolution transformer
# =============================================================================
#
# window_sizes recovery: RIB / RIBSiren / RoPEViT / NoPE / CPE attention encodes
# NOTHING window-size-dependent in the saved weights, so those models are not
# self-describing from weights alone. The current arch records a persistent
# `_window_sizes` buffer for exact recovery; these presets are only a best-effort
# fallback for older (pre-buffer) checkpoints. Keyed by (dim, n_blocks, n_stl).
# The +/non-+ variants are indistinguishable without the buffer — the fallback
# picks the non-"+" (standard) window set; re-save such a model to embed the
# buffer for an exact round-trip.
_KNOWN_PRESETS: dict[tuple[int, int, int], list[int]] = {
    (48, 5, 6): [8, 16, 32, 16, 32, 64],  # SSTlight
    (180, 6, 6): [16, 32, 64, 16, 32, 64],  # SST (base)
    (192, 8, 6): [16, 32, 64, 16, 32, 64],  # SSTLarge (SST-L)
    (224, 10, 6): [16, 32, 48, 32, 48, 96],  # SSTXL_Plus (SST-XL+, plus-only)
}


def _detect_attn_type(state_dict: StateDict, prefix: str) -> str:
    """Detect attention type from an STL block's attn.* keys under `prefix`."""
    if f"{prefix}flashbias_q" in state_dict:
        return "FlashBias"
    if f"{prefix}to_hidden" in state_dict:
        n_input = state_dict[f"{prefix}to_hidden"].shape[0]
        return "RIBSiren" if n_input == 2 else "RIB"
    if f"{prefix}rope_freqs" in state_dict:
        return "RoPEViT"
    if f"{prefix}relative_position_bias" in state_dict:
        # Naive / SDPA / Flex share weights; SDPA is the portable inference path.
        return "SDPA"
    return "NoPE"


def _detect_window_sizes_from_weights(
    state_dict: StateDict, attn_type: str, n_stl: int, body_prefix: str
) -> list[int] | None:
    """Recover per-STL window sizes directly from weights, when possible."""
    if attn_type == "FlashBias":
        ws = []
        for s in range(n_stl):
            key = f"{body_prefix}{s}.attn.flashbias_q"
            if key not in state_dict:
                return None
            ws.append(int(math.isqrt(state_dict[key].shape[1])))
        return ws
    if attn_type in ("SDPA", "Naive", "Flex"):
        ws = []
        for s in range(n_stl):
            key = f"{body_prefix}{s}.attn.relative_position_bias"
            if key not in state_dict:
                return None
            side = int(math.isqrt(state_dict[key].shape[1]))  # (2*ws-1)
            ws.append((side + 1) // 2)
        return ws
    return None  # RIB / RIBSiren / RoPEViT / NoPE: not recoverable from weights


def _detect_num_heads(
    state_dict: StateDict, attn_type: str, n_stl: int, dim: int, body_prefix: str
) -> list[int]:
    heads: list[int] = []
    for s in range(n_stl):
        prefix = f"{body_prefix}{s}.attn."
        if attn_type == "FlashBias" and f"{prefix}flashbias_q" in state_dict:
            heads.append(state_dict[f"{prefix}flashbias_q"].shape[0])
        elif attn_type in ("RIB", "RIBSiren") and f"{prefix}to_q" in state_dict:
            heads.append(state_dict[f"{prefix}to_q"].shape[0])
        elif attn_type == "RoPEViT" and f"{prefix}rope_freqs" in state_dict:
            heads.append(state_dict[f"{prefix}rope_freqs"].shape[1])
        elif f"{prefix}relative_position_bias" in state_dict:
            heads.append(state_dict[f"{prefix}relative_position_bias"].shape[0])
        else:
            for nh in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
                if dim % nh == 0:
                    heads.append(nh)
                    break
    if len(heads) != n_stl:
        raise ValueError("Could not detect num_heads for all STL blocks")
    return heads


def _detect_ranks(
    state_dict: StateDict, attn_type: str, n_stl: int, body_prefix: str
) -> list[int | None]:
    ranks: list[int | None] = []
    for s in range(n_stl):
        prefix = f"{body_prefix}{s}.attn."
        if attn_type == "FlashBias" and f"{prefix}flashbias_q" in state_dict:
            ranks.append(state_dict[f"{prefix}flashbias_q"].shape[2])
        elif attn_type in ("RIB", "RIBSiren") and f"{prefix}to_q" in state_dict:
            ranks.append(state_dict[f"{prefix}to_q"].shape[2])
        else:
            ranks.append(None)
    return ranks


def _detect_gate_type(state_dict: StateDict, prefix: str) -> str | None:
    if f"{prefix}gate.1.weight" in state_dict:
        return "DWC"
    if f"{prefix}gate.0.weight" in state_dict:
        return "Linear"
    return None


def _detect_rib_params(
    state_dict: StateDict, attn_type: str, prefix: str
) -> tuple[int | None, int | None]:
    if attn_type not in ("RIB", "RIBSiren"):
        return None, None
    to_hidden = state_dict[f"{prefix}to_hidden"]
    rib_hidden_dim = to_hidden.shape[1]
    n_input = to_hidden.shape[0]
    rib_n_freqs = (n_input - 2) // 4 if n_input > 2 else 0
    return rib_hidden_dim, rib_n_freqs


def _detect_upsampler(state_dict: StateDict) -> tuple[str, int, int]:
    if "upsampler.feature_conv.weight" in state_dict:
        intermediate_dim = state_dict["upsampler.feature_conv.weight"].shape[0]
        up_out = state_dict["upsampler.up_conv.weight"].shape[0]
        up_in = state_dict["upsampler.up_conv.weight"].shape[1]
        scale = int(round(math.sqrt(up_out / up_in)))
        return "pixelshuffle", scale, intermediate_dim
    if "upsampler.f.0.weight" in state_dict:
        intermediate_dim = state_dict["upsampler.f.0.weight"].shape[0]
        n_modules = get_seq_len(state_dict, "upsampler.f")
        n_ups = (n_modules - 3) // 3
        scale = 2**n_ups if n_ups > 0 else 1
        return "nn+conv", scale, intermediate_dim
    up_out = state_dict["upsampler.up_conv.weight"].shape[0]
    scale = int(round(math.sqrt(up_out / 3)))
    return "pixelshuffle_direct", scale, 64


class SSTArch(Architecture[SST]):
    def __init__(self) -> None:
        super().__init__(
            id="SST",
            detect=KeyCondition.has_all(
                "proj.weight",
                "proj.bias",
                "body.0.0.norm_attn.weight",
                "body.0.0.attn.to_qkv.weight",
                "body.0.0.attn.to_out.weight",
                "body.0.0.norm_ffn.weight",
                "body.0.0.ffn.proj.weight",
                "body.0.0.ffn.dwc.weight",
                "body.0.0.ffn.agg.weight",
                "upsampler.up_conv.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SST]:
        dim = state_dict["proj.weight"].shape[0]

        body_len = get_seq_len(state_dict, "body")
        n_blocks = body_len - 2  # last two entries are LayerNorm + Conv2d
        n_stl = get_seq_len(state_dict, "body.0") - 1  # last entry is the SSTB Conv2d

        attn_type = _detect_attn_type(state_dict, "body.0.0.attn.")

        # window_sizes: prefer the self-describing buffer; else recover from
        # weights (FlashBias / RPB types); else fall back to known presets.
        if "_window_sizes" in state_dict:
            window_sizes = [int(w) for w in state_dict["_window_sizes"].tolist()]
        else:
            recovered = _detect_window_sizes_from_weights(
                state_dict, attn_type, n_stl, "body.0."
            )
            if recovered is not None:
                window_sizes = recovered
            else:
                preset_key = (dim, n_blocks, n_stl)
                if preset_key not in _KNOWN_PRESETS:
                    raise ValueError(
                        f"Cannot recover window_sizes for SST (dim={dim}, "
                        f"n_blocks={n_blocks}, n_stl={n_stl}, attn_type={attn_type}): "
                        f"no _window_sizes buffer and no matching preset. Re-save the "
                        f"model with the current arch to embed the buffer."
                    )
                window_sizes = _KNOWN_PRESETS[preset_key]
        # Keep _window_sizes in the state dict: the model has the buffer and will
        # load it. (The arch also injects it when absent, for old checkpoints.)

        num_heads = _detect_num_heads(state_dict, attn_type, n_stl, dim, "body.0.")
        ranks = _detect_ranks(state_dict, attn_type, n_stl, "body.0.")

        ffn_hidden = state_dict["body.0.0.ffn.proj.weight"].shape[0]
        exp_ratio = ffn_hidden / dim

        rib_hidden_dim, rib_n_freqs = _detect_rib_params(
            state_dict, attn_type, "body.0.0.attn."
        )
        gate_type = _detect_gate_type(state_dict, "body.0.0.attn.")
        upsampler_type, upscaling_factor, intermediate_dim = _detect_upsampler(
            state_dict
        )

        model = SST(
            dim=dim,
            window_sizes=window_sizes,
            num_heads=num_heads,
            n_blocks=n_blocks,
            exp_ratio=exp_ratio,
            attn_type=attn_type,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            upscaling_factor=upscaling_factor,
            upsampler_type=upsampler_type,
            ranks=ranks,
            gate_type=gate_type,
            intermediate_dim=intermediate_dim,
        )

        tags = [
            f"{dim}dim",
            f"{n_blocks}blocks",
            f"{n_stl}stl",
            attn_type,
        ]
        if gate_type:
            tags.append(f"gate={gate_type}")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscaling_factor == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=upscaling_factor,
            input_channels=3,
            output_channels=3,
            # SST reflect-pads each input up to a multiple of the window size, and
            # reflect padding requires the pad to be smaller than the input. The
            # input must therefore be at least as large as the largest window, or
            # small tiles will crash in pad_to_win.
            size_requirements=SizeRequirements(minimum=max(window_sizes)),
        )


# =============================================================================
# SST-Real — single-step EDM diffusion model (EDMUNet), wrapped for chaiNNer
# =============================================================================


def _detect_real_exp_ratio(dim: int, w12_out: int) -> float:
    """Recover exp_ratio for a RealSTL ConvSwiGLUFFN.

    ConvSwiGLUFFN: hidden = int(int(dim * exp_ratio) * 2 / 3); w12.0 out = 2 * hidden.
    Search nice ratios that reproduce w12_out exactly, else the closest computed value.
    """
    target_hidden = w12_out // 2
    for r in (3.0, 2.0, 2.667, 4.0, 1.5, 2.5, 3.5, 1.25):
        if int(int(dim * r) * 2 / 3) == target_hidden:
            return r
    return target_hidden * 3 / (2 * dim)


class SSTRealArch(Architecture[EDMUNetSR]):
    """SST-Real (EDMUNet): a single-step EDM diffusion SR model.

    The saved checkpoint is an ``EDMUNet`` whose only weights live under
    ``model.*`` (the ``SSTReal`` core). It is loaded into ``EDMUNetSR``, a
    deterministic single-step-sampling wrapper that exposes ``forward(lq)->sr``.
    """

    def __init__(self) -> None:
        super().__init__(
            id="SSTReal",
            detect=KeyCondition.has_all(
                "model.proj_in.weight",
                "model.proj_lr.weight",
                "model.map_noise.mlp.0.weight",
                "model.map_noise.mlp.2.weight",
                "model.proj_body.0.adaLN_modulation.weight",
                "model.proj_body.0.out_conv.weight",
                "model.up_i.0.weight",
                "model.up_f.0.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[EDMUNetSR]:
        p = "model."
        bp = f"{p}proj_body."

        dim = state_dict[f"{p}proj_in.weight"].shape[0]
        num_blocks = get_seq_len(state_dict, f"{p}proj_body")
        n_stl = get_seq_len(state_dict, f"{bp}0.blocks")

        emb_channels = state_dict[f"{p}map_noise.mlp.0.weight"].shape[0]
        noise_channels = state_dict[f"{p}map_noise.mlp.0.weight"].shape[1]
        intermediate_dim = state_dict[f"{p}up_i.0.weight"].shape[0]

        blk0_attn = f"{bp}0.blocks.0.attn."
        attn_type = _detect_attn_type(state_dict, blk0_attn)

        # window_sizes: SSTReal is not self-describing; use the known preset unless
        # a future checkpoint embeds a _window_sizes buffer.
        if f"{p}_window_sizes" in state_dict:
            window_sizes = [int(w) for w in state_dict[f"{p}_window_sizes"].tolist()]
        else:
            recovered = _detect_window_sizes_from_weights(
                state_dict, attn_type, n_stl, f"{bp}0.blocks."
            )
            window_sizes = (
                recovered if recovered is not None else [16, 32, 64, 16, 32, 64]
            )

        num_heads = _detect_num_heads(
            state_dict, attn_type, n_stl, dim, f"{bp}0.blocks."
        )
        ranks = _detect_ranks(state_dict, attn_type, n_stl, f"{bp}0.blocks.")
        rib_hidden_dim, rib_n_freqs = _detect_rib_params(
            state_dict, attn_type, blk0_attn
        )
        gate_type = _detect_gate_type(state_dict, blk0_attn)

        w12_out = state_dict[f"{bp}0.blocks.0.ffn.w12.0.weight"].shape[0]
        exp_ratio = _detect_real_exp_ratio(dim, w12_out)

        # fuse_lr present on blocks 0..fuse_lr_till
        fuse_lr_till = -1
        for i in range(num_blocks):
            if f"{bp}{i}.fuse_lr.weight" in state_dict:
                fuse_lr_till = i
        # ig_idx is a forward-time control, not stored. It matches fuse_lr_till in
        # the reference config; use that as the best available estimate.
        ig_idx = fuse_lr_till if fuse_lr_till >= 0 else num_blocks // 2

        model = EDMUNetSR(
            scale=4,
            sigma_data=0.5,
            sigma_max=1.0,
            use_skip=True,
            sampling_ig_lambda=1.0,
            min_size=64,
            seed=0,
            dim=dim,
            intermediate_dim=intermediate_dim,
            emb_channels=emb_channels,
            noise_channels=noise_channels,
            num_blocks=num_blocks,
            fuse_lr_till=fuse_lr_till,
            ig_idx=ig_idx,
            attn_type=attn_type,
            window_sizes=window_sizes,
            num_heads=num_heads,
            ranks=ranks,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            exp_ratio=exp_ratio,
            gate_type=gate_type,
            drop_p=0.0,
        )

        tags = [
            f"{dim}dim",
            f"{num_blocks}blocks",
            f"{n_stl}stl",
            attn_type,
            "1-step-diffusion",
        ]
        if gate_type:
            tags.append(f"gate={gate_type}")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR",
            tags=tags,
            supports_half=False,
            supports_bfloat16=True,
            scale=4,
            input_channels=3,
            output_channels=3,
            size_requirements=SizeRequirements(minimum=64),
        )


__all__ = ["SSTArch", "SSTRealArch", "SST", "EDMUNetSR"]
