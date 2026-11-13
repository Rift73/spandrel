from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.heart_arch import HEART

_BLOCK = "layers.0.residual_group.blocks.0."


class HEARTArch(Architecture[HEART]):
    def __init__(self) -> None:
        super().__init__(
            id="HEART",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "layers.0.residual_group.blocks.0.norm1.weight",
                "layers.0.residual_group.blocks.0.attn.to_qkv.weight",
                "layers.0.residual_group.blocks.0.attn.to_out.weight",
                "layers.0.residual_group.blocks.0.attn.to_hidden",
                "layers.0.residual_group.blocks.0.attn.hidden_b",
                "layers.0.residual_group.blocks.0.attn.to_q",
                "layers.0.residual_group.blocks.0.attn.to_k",
                "layers.0.residual_group.blocks.0.conv_block.cab.0.weight",
                "layers.0.residual_group.blocks.0.conv_block.cab.3.attention.1.weight",
                "layers.0.residual_group.blocks.0.mlp.fc1.weight",
                "layers.0.residual_group.blocks.0.norm2.weight",
                "norm.weight",
                "conv_before_upsample.0.weight",
                "conv_last.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[HEART]:
        embed_dim, in_chans = state_dict["conv_first.weight"].shape[:2]
        num_feat = state_dict["conv_before_upsample.0.weight"].shape[0]

        num_layers = get_seq_len(state_dict, "layers")
        depths = [
            get_seq_len(state_dict, f"layers.{i}.residual_group.blocks")
            for i in range(num_layers)
        ]
        num_heads = [
            state_dict[f"layers.{i}.residual_group.blocks.0.attn.to_q"].shape[0]
            for i in range(num_layers)
        ]

        # attention runs in blocks i % attention_freq == 0
        attention_blocks = [
            i
            for i in range(depths[0])
            if f"layers.0.residual_group.blocks.{i}.attn.to_q" in state_dict
        ]
        attention_freq = (
            attention_blocks[1] if len(attention_blocks) > 1 else max(2, depths[0])
        )

        n_input, rib_hidden_dim = state_dict[_BLOCK + "attn.to_hidden"].shape
        rib_n_freqs = (n_input - 2) // 4
        rank = state_dict[_BLOCK + "attn.to_q"].shape[2]

        compress_ratio = (
            embed_dim // state_dict[_BLOCK + "conv_block.cab.0.weight"].shape[0]
        )
        squeeze_factor = (
            embed_dim
            // state_dict[_BLOCK + "conv_block.cab.3.attention.1.weight"].shape[0]
        )
        mlp_ratio = state_dict[_BLOCK + "mlp.fc1.weight"].shape[0] / embed_dim

        resi_connection = (
            "1conv" if "conv_after_body.weight" in state_dict else "identity"
        )
        patch_norm = "patch_embed.norm.weight" in state_dict

        ape = "absolute_pos_embed" in state_dict
        img_size = 64
        if ape:
            # patch_size 1: one patch per pixel of a square img_size crop
            num_patches = state_dict["absolute_pos_embed"].shape[1]
            img_size = round(num_patches**0.5)

        # pixelshuffle: a conv and a PixelShuffle(2) per 2x stage, or a single
        # 9x conv for 3x; scale 1 has no upsample layers
        if "upsample.0.weight" not in state_dict:
            upscale = 1
        elif state_dict["upsample.0.weight"].shape[0] == 9 * num_feat:
            upscale = 3
        else:
            upscale = 2 ** ((get_seq_len(state_dict, "upsample") + 1) // 2)

        # window_size is not stored in the weights (the RIB coordinates are a
        # non-persistent buffer); every released model uses 32
        window_size = 32

        model = HEART(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            mlp_ratio=mlp_ratio,
            ape=ape,
            patch_norm=patch_norm,
            upscale=upscale,
            resi_connection=resi_connection,
            num_feat=num_feat,
            rank=rank,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            attention_freq=attention_freq,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=[f"{embed_dim}dim", f"{num_layers}x{depths[0]}", f"r{rank}"],
            supports_half=True,
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_chans,
            output_channels=in_chans,
            # The forward pads to the window itself, choosing reflect or replicate by
            # the input's size, a choice a trace freezes: an exported model takes no
            # side smaller than a window.
            size_requirements=SizeRequirements(minimum=window_size),
        )


__all__ = ["HEARTArch", "HEART"]
