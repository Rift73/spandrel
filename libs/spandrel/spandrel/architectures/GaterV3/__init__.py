from __future__ import annotations

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


def _get_upsampler(state_dict: StateDict) -> tuple[int, SampleMods, int, int, int]:
    """The scale, upsampler, upsample_mid_dim, end_kernel and MetaUpsample version
    of a model with scale > 1."""
    # The model cannot load without this buffer (strict loading). Version 1 is
    # gaterv3.2's UniUpsample, version 3 resselt's gaterv3.3; both write
    # upsample_mid_dim, which shapes no layer for some upsamplers, and the scale,
    # which is the only difference between nearest+conv at x2 and at x3.
    version, index, scale, _, _, mid_dim, _ = (
        int(value) for value in state_dict[_META].tolist()
    )
    upsample: SampleMods = list(SampleMods.__args__)[index]
    if upsample == "conv":
        # one conv at the input size: its output cannot be added to the
        # upscaled input, so no such model can run
        raise ValueError(f"GaterV3's conv upsampler cannot upscale {scale}x")

    end_kernel = 1
    for key in ("dim_to_in.0.end_conv.weight", "dim_to_in.2.end_conv.weight"):
        if key in state_dict:
            # dysample's last conv, after the conv in front of it if any
            end_kernel = state_dict[key].shape[2]
    return scale, upsample, mid_dim, end_kernel, version


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
                # scale 1 ends in a plain conv, higher scales in an upsampler
                KeyCondition.has_any("dim_to_in.weight", _META),
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
        # and is kept in a buffer; the first gaterv3.py had no buffer and always
        # ran channel attention. Without attention it leaves no trace (default).
        flash = True
        if attention:
            flash = "latent.0.token_mix.flash" in state_dict and bool(
                state_dict["latent.0.token_mix.flash"].item()
            )

        if "dim_to_in.weight" in state_dict:
            # scale 1 ends in a plain conv: the upsampler's options leave no
            # trace and keep their defaults
            scale = 1
            upsample: SampleMods = "pixelshuffle"
            upsample_mid_dim = 48
            end_kernel = 1
            upsample_version = 1
        else:
            scale, upsample, upsample_mid_dim, end_kernel, upsample_version = (
                _get_upsampler(state_dict)
            )

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
            end_kernel=end_kernel,
            upsample_version=upsample_version,
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


__all__ = ["GaterV3Arch", "GateRV3"]
