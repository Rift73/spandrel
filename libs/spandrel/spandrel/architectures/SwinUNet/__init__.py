from __future__ import annotations

import math

from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_seq_len
from .__arch.swin_unet import SwinUNet, SwinUNetBase


class SwinUNetArch(Architecture[SwinUNet]):
    def __init__(self) -> None:
        super().__init__(
            id="SwinUNet",
            detect=KeyCondition.has_all(
                "unet.patch.0.weight",
                "unet.patch.2.weight",
                "unet.swin1.block.0.attn.qkv.weight",
                "unet.swin1.block.0.attn.relative_position_bias_table",
                "unet.swin1.block.0.attn.relative_position_index",
                "unet.down1.conv.weight",
                "unet.up1.proj.weight",
                "unet.up2.proj.weight",
                KeyCondition.has_any(
                    "unet.to_image.proj.weight",  # 1x/2x/4x
                    "unet.to_image.proj.0.weight",  # 8x (Linear-LReLU-Linear)
                ),
            ),
        )

    @staticmethod
    def _count_blocks(state_dict: StateDict, prefix: str) -> int:
        return get_seq_len(state_dict, prefix)

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SwinUNet]:
        # ---- stem ----
        in_channels = state_dict["unet.patch.0.weight"].shape[1]
        base_dim = state_dict["unet.patch.2.weight"].shape[0]

        # ---- attention hyper-parameters ----
        bias_table = state_dict["unet.swin1.block.0.attn.relative_position_bias_table"]
        num_heads = bias_table.shape[1]
        window = (math.isqrt(bias_table.shape[0]) + 1) // 2
        assert (2 * window - 1) ** 2 == bias_table.shape[0], (
            f"unexpected relative_position_bias_table shape {tuple(bias_table.shape)}"
        )

        # ---- depth ----
        base_layers = self._count_blocks(state_dict, "unet.swin1.block")
        assert self._count_blocks(state_dict, "unet.swin2.block") == base_layers
        assert self._count_blocks(state_dict, "unet.swin4.block") == base_layers
        assert self._count_blocks(state_dict, "unet.swin5.block") == base_layers
        assert self._count_blocks(state_dict, "unet.swin3.block") == 3 * base_layers
        # nunif asserts base_layers % 2 == 0
        assert base_layers % 2 == 0, f"base_layers={base_layers} is not even"

        # ---- scale / decoder variant ----
        wide_decoder = "unet.proj2.weight" in state_dict  # Linear(C -> 2C), 4x/8x only
        if "unet.to_image.proj.weight" in state_dict:
            out_total = state_dict["unet.to_image.proj.weight"].shape[0]
        else:  # 8x head: proj.0 -> proj.2
            out_total = state_dict["unet.to_image.proj.2.weight"].shape[0]
        scale = math.isqrt(out_total // 3)
        assert out_total == 3 * scale**2, (
            f"cannot derive a scale factor from to_image output dim {out_total}"
        )
        if wide_decoder:
            assert scale in {4, 8}, (scale, "proj2 present but scale is not 4x/8x")
        else:
            assert scale in {1, 2}, (scale, "proj2 absent but scale is not 1x/2x")

        model = SwinUNet(
            unet=SwinUNetBase(
                in_channels=in_channels,
                out_channels=3,
                base_dim=base_dim,
                base_layers=base_layers,
                scale_factor=scale,
            ),
            scale=scale,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[
                f"{base_dim}dim",
                f"{base_layers * 2 + base_layers * 3}blocks",
                f"window{window}",
            ],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_channels,
            output_channels=3,
        )


__all__ = ["SwinUNetArch", "SwinUNet", "SwinUNetBase"]
