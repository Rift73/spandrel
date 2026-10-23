from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.figsr_arch import FIGSR, SampleMods


class FIGSRArch(Architecture[FIGSR]):
    def __init__(self) -> None:
        super().__init__(
            id="FIGSR",
            detect=KeyCondition.has_all(
                "in_to_dim.weight",
                # gfisr_body_half is empty for a single block
                "gfisr_body_half_2.0.norm.scale",
                "gfisr_body_half_2.0.norm.rms",
                "gfisr_body_half_2.0.fc1.weight",
                "gfisr_body_half_2.0.conv.fu.rn.scale",
                "gfisr_body_half_2.0.conv.fu.post_norm.scale",
                "gfisr_body_half_2.0.conv.fu.fdc.weight",
                "gfisr_body_half_2.0.conv.fu.fpe.weight",
                "gfisr_body_half_2.0.conv.convhw.weight",
                "gfisr_body_half_2.0.conv.convw.weight",
                "gfisr_body_half_2.0.conv.convh.weight",
                "gfisr_body_half_2.0.fc2.weight",
                "cat_to_dim.weight",
                "shift",
                "scale_norm",
                "upscale.MetaUpsample",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[FIGSR]:
        # version, upsampler index, scale, in_dim, out_dim, mid_dim, group
        _, index, scale, _, out_nc, mid_dim, _ = (
            int(value) for value in state_dict["upscale.MetaUpsample"].tolist()
        )
        upsampler = list(SampleMods.__args__)[index]
        dim, in_nc = state_dict["in_to_dim.weight"].shape[:2]

        # the second half holds the odd block and ends with a conv
        n_blocks = (
            get_seq_len(state_dict, "gfisr_body_half")
            + get_seq_len(state_dict, "gfisr_body_half_2")
            - 1
        )
        block = "gfisr_body_half_2.0."
        hidden = state_dict[block + "fc1.weight"].shape[0] // 2
        # hidden = int(expansion_ratio * dim) // 8 * 8: the default 8/3 when it fits
        expansion_ratio = next(
            (r for r in (8 / 3, hidden / dim) if int(r * dim) // 8 * 8 == hidden),
            (hidden + 0.5) / dim,
        )
        gc, _, square_kernel_size, _ = state_dict[block + "conv.convhw.weight"].shape
        band_kernel_size = state_dict[block + "conv.convw.weight"].shape[3]

        model = FIGSR(
            in_nc=in_nc,
            dim=dim,
            expansion_ratio=expansion_ratio,
            scale=scale,
            out_nc=out_nc,
            upsampler=upsampler,
            mid_dim=mid_dim,
            n_blocks=n_blocks,
            gc=gc,
            square_kernel_size=square_kernel_size,
            band_kernel_size=band_kernel_size,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}dim", f"{n_blocks}nb", upsampler],
            # FourierUnit runs its FFTs in fp32 whatever the model's dtype
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_nc,
            output_channels=out_nc,
            # reflect padding to an even size needs two pixels
            size_requirements=SizeRequirements(minimum=2),
        )


__all__ = ["FIGSRArch", "FIGSR"]
