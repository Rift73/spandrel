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
from .__arch.gaterv3 import GateRV3, SampleMods

# UniUpsample records itself in this uint8 buffer:
# [version, upsampler index, scale, in_dim, out_dim, mid_dim, group]
_META = "dim_to_in.MetaUpsample"


def _get_upsampler(state_dict: StateDict, in_ch: int) -> tuple[int, SampleMods, int]:
    """The scale, upsampler and upsample_mid_dim of a model with scale > 1."""
    # The model cannot load without this buffer (strict loading).
    meta = state_dict[_META]
    # upsample_mid_dim only shapes the pixelshuffle upsampler's layers; the
    # buffer records it for the others
    mid_dim = int(meta[5])
    convs = sum(f"dim_to_in.{i}.weight" in state_dict for i in range(32))

    upsample: SampleMods
    if "dim_to_in.0.offset.weight" in state_dict:
        upsample = "dysample"
        # 2 * groups (4) * scale**2 offsets
        scale = math.isqrt(state_dict["dim_to_in.0.offset.weight"].shape[0] // 8)
    elif "dim_to_in.2.weight" in state_dict:
        upsample = "pixelshuffle"
        mid_dim = state_dict["dim_to_in.0.weight"].shape[0]
        if state_dict["dim_to_in.2.weight"].shape[0] == 9 * mid_dim:
            scale = 3
        else:
            # in conv, one conv per x2 step, out conv
            scale = 2 ** (convs - 2)
    elif "dim_to_in.3.weight" in state_dict:
        upsample = "nearest+conv"
        # x2 and x3 have the same layers; only the buffer tells them apart
        scale = int(meta[2])
    else:
        upsample = "pixelshuffledirect"
        scale = math.isqrt(state_dict["dim_to_in.0.weight"].shape[0] // in_ch)
    return scale, upsample, mid_dim


class GaterV3Arch(Architecture[GateRV3]):
    def __init__(self) -> None:
        super().__init__(
            id="GaterV3",
            detect=KeyCondition.has_all(
                "in_to_dim.weight",
                "in_to_dim.bias",
                "span_block0.c1_r.sk.weight",
                "span_block0.c1_r.conv.0.weight",
                "span_block0.c3_r.eval_conv.weight",
                "span_end.c1_r.sk.weight",
                "span_end.c3_r.eval_conv.weight",
                "sisr_end_conv.sk.weight",
                "sisr_end_conv.eval_conv.weight",
                "sisr_cat_conv.weight",
                "sisr_cat_conv.bias",
                KeyCondition.has_any(
                    "dim_to_in.weight",
                    "dim_to_in.0.weight",
                    "dim_to_in.0.offset.weight",
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[GateRV3]:
        in_ch = state_dict["in_to_dim.weight"].shape[1]
        dim = state_dict["in_to_dim.weight"].shape[0]
        enc_blocks = tuple(
            get_seq_len(state_dict, f"gater_encode.{i}.gated")
            for i in range(get_seq_len(state_dict, "gater_encode"))
        )
        dec_blocks = tuple(
            get_seq_len(state_dict, f"decode.{i}.gated")
            for i in range(get_seq_len(state_dict, "decode"))
        )
        num_latent = get_seq_len(state_dict, "latent")
        sisr_blocks = get_seq_len(state_dict, "span_n_b")

        attention = "latent.0.token_mix.qkv.weight" in state_dict
        # flash chooses the attention's math (spatial SDPA or channel attention)
        # and is kept in a buffer; without attention it leaves no trace (default)
        flash = True
        if "latent.0.token_mix.flash" in state_dict:
            flash = bool(state_dict["latent.0.token_mix.flash"].item())

        if "dim_to_in.weight" in state_dict:
            # scale 1 ends in a plain conv: upsample and upsample_mid_dim
            # leave no trace and keep their defaults
            scale = 1
            upsample: SampleMods = "pixelshuffle"
            upsample_mid_dim = 48
        else:
            scale, upsample, upsample_mid_dim = _get_upsampler(state_dict, in_ch)

        # end_gamma_init only sets gamma's initial value, which the weights replace
        model = GateRV3(
            in_ch=in_ch,
            dim=dim,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_latent=num_latent,
            scale=scale,
            upsample=upsample,
            upsample_mid_dim=upsample_mid_dim,
            attention=attention,
            sisr_blocks=sisr_blocks,
            flash=flash,
        )

        tags = [f"{dim}dim"]
        if scale != 1:
            tags.append(upsample)
        if attention:
            tags.append("attention")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_ch,
            output_channels=in_ch,
            # the U-Net halves the size once per encoder level; the model
            # reflect-pads to this multiple itself, as spandrel does
            size_requirements=SizeRequirements(multiple_of=2 ** len(enc_blocks)),
        )


GaterV3 = GateRV3

__all__ = ["GaterV3Arch", "GateRV3", "GaterV3"]
