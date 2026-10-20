from __future__ import annotations

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from spandrel.util import KeyCondition, get_seq_len

from .__arch.gaterv3 import GaterV3, SampleMods


class GaterV3Arch(Architecture[GaterV3]):
    def __init__(self):
        super().__init__(
            id="GaterV3",
            detect=KeyCondition.has_all(
                "in_to_dim.weight",
                "gater_encode.0.gated.0.local.1.weight",
                "span_block0.c1_r.sk.weight",
                "sisr_cat_conv.weight",
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[GaterV3]:
        # Extract basic parameters
        in_ch = state_dict["in_to_dim.weight"].shape[1]
        dim = state_dict["in_to_dim.weight"].shape[0]
        
        # Extract encoder/decoder structure
        num_enc_blocks = get_seq_len(state_dict, "gater_encode")
        enc_blocks = []
        for i in range(num_enc_blocks):
            num_gated = get_seq_len(state_dict, f"gater_encode.{i}.gated")
            enc_blocks.append(num_gated)
        
        num_dec_blocks = get_seq_len(state_dict, "decode")
        dec_blocks = []
        for i in range(num_dec_blocks):
            num_gated = get_seq_len(state_dict, f"decode.{i}.gated")
            dec_blocks.append(num_gated)
        
        # Extract latent blocks
        num_latent = get_seq_len(state_dict, "latent")
        
        # Extract SISR blocks
        sisr_blocks = get_seq_len(state_dict, "span_n_b")
        
        # Determine scale and upsampler type
        scale = 1
        upsample = "conv"
        upsample_mid_dim = 48
        
        if "dim_to_in.MetaUpsample" in state_dict:
            meta_upsample = state_dict["dim_to_in.MetaUpsample"].tolist()
            upsample_index = meta_upsample[1]
            scale = meta_upsample[2]
            upsample_mid_dim = meta_upsample[5]
            upsample = list(SampleMods.__args__)[upsample_index]
        elif hasattr(state_dict.get("short_to_dim", {}), "scale_factor"):
            # Fallback: try to extract scale from upsampler
            scale = int(state_dict["short_to_dim.scale_factor"])
        
        # Extract gamma initialization value
        if "gamma" in state_dict:
            end_gamma_init = int(state_dict["gamma"].mean().item())
        else:
            end_gamma_init = 1
        
        # Determine if attention is used (check if any latent block has attention)
        attention = False
        if f"latent.0.token_mix.qkv.weight" in state_dict:
            attention = True
        
        # Flash attention (default to True for newer models)
        flash = True
        
        model = GaterV3(
            in_ch=in_ch,
            dim=dim,
            enc_blocks=tuple(enc_blocks),
            dec_blocks=tuple(dec_blocks),
            num_latent=num_latent,
            scale=scale,
            upsample=upsample,
            upsample_mid_dim=upsample_mid_dim,
            end_gamma_init=end_gamma_init,
            attention=attention,
            sisr_blocks=sisr_blocks,
            flash=flash,
        )

        # Determine size tags
        size_tags = []
        if enc_blocks == [2, 2, 4] and dec_blocks == [2, 2, 2]:
            size_tags.append("small")
        elif enc_blocks == [2, 2, 4, 6] and dec_blocks == [2, 2, 2, 2]:
            size_tags.append("regular")
        
        tags = [
            f"{dim}dim",
            f"{num_latent}latent",
            upsample,
        ]
        tags.extend(size_tags)
        
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
            size_requirements=SizeRequirements(multiple_of=model.pad),
        )


__all__ = ["GaterV3Arch", "GaterV3"]
