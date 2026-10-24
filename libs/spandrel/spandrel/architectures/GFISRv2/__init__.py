from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.gfisrv2 import GFISRV2, SampleMods


class GFISRv2Arch(Architecture[GFISRV2]):
    def __init__(self) -> None:
        super().__init__(
            id="GFISRv2",
            detect=KeyCondition.has_all(
                KeyCondition.has_any("in_to_dim.weight", "in_to_dim.1.weight"),
                "gfisr_body.0.gamma",
                "gfisr_body.0.norm.scale",
                "gfisr_body.0.norm.offset",
                "gfisr_body.0.fc1.weight",
                # block i's Fourier unit takes inception slot i % 4: pconv first
                "gfisr_body.0.conv.pconv.rn.scale",
                "gfisr_body.0.conv.pconv.post_norm.scale",
                "gfisr_body.0.conv.pconv.fdc.weight",
                "gfisr_body.0.conv.pconv.fpe.weight",
                "gfisr_body.0.conv.dwconv_hw.weight",
                "gfisr_body.0.conv.dwconv_w.weight",
                "gfisr_body.0.conv.dwconv_h.weight",
                "gfisr_body.0.fc2.weight",
                "upscale.MetaUpsample",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[GFISRV2]:
        # version, upsampler index, scale, in_dim, out_dim, mid_dim, group
        _, index, upscale, _, out_nc, mid_dim, _ = (
            int(value) for value in state_dict["upscale.MetaUpsample"].tolist()
        )
        upsampler = list(SampleMods.__args__)[index]

        pixel_unshuffle = "in_to_dim.1.weight" in state_dict
        if pixel_unshuffle:
            # At scales 1 and 2 the input is unshuffled by 4 // scale and the
            # upsampler runs at 4x. The weights only hold in_nc * (4 // scale)**2:
            # take the reading with in_nc == out_nc, else 1x when the count
            # allows it (resselt's reading), else 2x. Some counts fit both
            # (16 = 4 channels at 2x or 1 channel at 1x).
            dim, unshuffled = state_dict["in_to_dim.1.weight"].shape[:2]
            readings = [
                (s, unshuffled // (4 // s) ** 2)
                for s in (1, 2)
                if unshuffled % (4 // s) ** 2 == 0
            ]
            if not readings:
                raise ValueError(
                    f"in_to_dim.1 takes {unshuffled} channels: not in_nc * 4 or 16"
                )
            scale, in_nc = next((r for r in readings if r[1] == out_nc), readings[0])
        else:
            dim, in_nc = state_dict["in_to_dim.weight"].shape[:2]
            scale = upscale

        # the blocks, then conv, SiLU, conv
        n_blocks = get_seq_len(state_dict, "gfisr_body") - 3
        hidden = state_dict["gfisr_body.0.fc1.weight"].shape[0] // 2
        # hidden = int(expansion_ratio * dim): the default 8/3 when it fits
        expansion_ratio = next(
            (r for r in (8 / 3, hidden / dim) if int(r * dim) == hidden),
            (hidden + 0.5) / dim,
        )

        model = GFISRV2(
            in_nc=in_nc,
            dim=dim,
            expansion_ratio=expansion_ratio,
            scale=scale,
            out_nc=out_nc,
            upsampler=upsampler,
            mid_dim=mid_dim,
            pixel_unshuffle=pixel_unshuffle,
            n_blocks=n_blocks,
        )

        tags = [f"{dim}dim", f"{n_blocks}nb", upsampler]
        if pixel_unshuffle:
            tags.append("unshuffle")
            # The model pads to a multiple of 2 * (4 // scale) but crops its
            # output for 4x, so it needs whole multiples of that padding.
            pad = 2 * (4 // scale)
            size_requirements = SizeRequirements(minimum=pad, multiple_of=pad)
        else:
            # reflect padding to an even size needs two pixels
            size_requirements = SizeRequirements(minimum=2)

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=tags,
            # FourierUnit runs its FFTs in fp32 whatever the model's dtype
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=size_requirements,
        )


__all__ = ["GFISRv2Arch", "GFISRV2"]
