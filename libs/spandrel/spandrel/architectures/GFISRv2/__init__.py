"""
Architecture definition for GFISRV2.
"""

from __future__ import annotations

# pylint: disable=relative-beyond-top-level, import-error
# pylint: disable=too-few-public-methods, too-many-locals

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.gfisrv2 import GFISRV2


class GFISRV2Arch(Architecture[GFISRV2]):
    """GFISRV2 Architecture descriptor."""

    def __init__(self) -> None:
        super().__init__(
            id="GFISRV2",
            detect=KeyCondition.has_all(
                "gfisr_body.0.gamma",
                "gfisr_body.0.norm.scale",
                "gfisr_body.0.norm.offset",
                "gfisr_body.0.fc1.weight",
                "gfisr_body.0.fc1.bias",
                "gfisr_body.0.conv.pconv.rn.scale",
                "gfisr_body.0.conv.pconv.rn.offset",
                "gfisr_body.0.conv.pconv.post_norm.scale",
                "gfisr_body.0.conv.pconv.post_norm.offset",
                "gfisr_body.0.conv.pconv.fdc.weight",
                "gfisr_body.0.conv.pconv.fdc.bias",
                "gfisr_body.0.conv.pconv.fpe.weight",
                "gfisr_body.0.conv.pconv.fpe.bias",
                "gfisr_body.0.conv.dwconv_hw.weight",
                "gfisr_body.0.conv.dwconv_hw.bias",
                "gfisr_body.0.conv.dwconv_w.weight",
                "gfisr_body.0.conv.dwconv_w.bias",
                "gfisr_body.0.conv.dwconv_h.weight",
                "gfisr_body.0.conv.dwconv_h.bias",
                "gfisr_body.0.fc2.weight",
                "gfisr_body.0.fc2.bias",
                "upscale.MetaUpsample",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[GFISRV2]:
        """Load a GFISRV2 model from a state dict."""
        samplemods = [
            "conv",
            "pixelshuffledirect",
            "pixelshuffle",
            "nearest+conv",
            "dysample",
            "transpose+conv",
            "lda",
            "pa_up",
        ]

        meta = state_dict["upscale.MetaUpsample"].tolist()
        upsampler_idx = int(meta[1])
        # Масштаб, сохраненный в MetaUpsample, принадлежит апсемплеру.
        upsampler_scale = int(meta[2])
        dim = int(meta[3])
        out_ch = int(meta[4])
        mid_dim = int(meta[5])

        upsampler = samplemods[upsampler_idx]
        n_blocks = get_seq_len(state_dict, "gfisr_body") - 3

        expansion_ratio = float(state_dict["gfisr_body.0.fc1.weight"].shape[0] / 2 / dim)

        if "in_to_dim.weight" in state_dict:
            pixel_unshuffle = False
            in_nc = state_dict["in_to_dim.weight"].shape[1]
            true_scale = upsampler_scale
        else:
            pixel_unshuffle = True
            conv_in_nc = state_dict["in_to_dim.1.weight"].shape[1]

            # Более надежный способ определить true_scale при использовании PixelUnshuffle.
            # В SR-моделях in_nc обычно равно out_ch.
            # Если down=4 (scale=1), conv_in_nc = in_nc * 16
            # Если down=2 (scale=2), conv_in_nc = in_nc * 4
            if conv_in_nc == out_ch * 16:
                in_nc = out_ch
                true_scale = 1
            elif conv_in_nc == out_ch * 4:
                in_nc = out_ch
                true_scale = 2
            else:
                # Откат к старой логике (на случай, если in_nc != out_ch)
                if conv_in_nc % 16 == 0:
                    in_nc = conv_in_nc // 16
                    true_scale = 1
                else:
                    in_nc = conv_in_nc // 4
                    true_scale = 2

        model = GFISRV2(
            in_nc=in_nc,
            dim=dim,
            expansion_ratio=expansion_ratio,
            scale=true_scale,
            out_nc=out_ch,
            upsampler=upsampler,
            mid_dim=mid_dim,
            pixel_unshuffle=pixel_unshuffle,
            n_blocks=n_blocks,
        )

        tags = [
            f"{dim}dim",
            f"{n_blocks}nb",
            upsampler,
        ]
        if pixel_unshuffle:
            tags.append("pixel_unshuffle")

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if true_scale == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=true_scale,
            input_channels=in_nc,
            output_channels=out_ch,
            size_requirements=SizeRequirements(
                minimum=4,
            ),
        )


__all__ = ["GFISRV2Arch", "GFISRV2"]
