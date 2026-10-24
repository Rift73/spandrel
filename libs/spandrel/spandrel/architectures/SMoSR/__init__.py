from __future__ import annotations

import math
from typing import get_args

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.smosr_arch import SampleMods, SMoSR

# UniUpsampleV4_light records itself in this uint8 buffer:
# [version, upsampler index, scale, in_dim, out_dim, mid_dim, group, rep]
_META = "upsampler.MetaUpsample"
_UPSAMPLERS: tuple[SampleMods, ...] = get_args(SampleMods)


def _get_upsampler(
    state_dict: StateDict, scale: int
) -> tuple[SampleMods, int, int, int]:
    """The upsampler, out_ch, upsampler_mid_dim and d_kernel."""
    # The model cannot load without this buffer (strict loading). It records
    # the upsampler (its only trace at scale 1, and what tells "conv" from
    # pixelshuffledirect) and mid_dim, which only some upsamplers' layers show.
    meta = state_dict[_META]
    upsampler = _UPSAMPLERS[int(meta[1])]
    mid_dim = int(meta[5])
    d_kernel = 3  # SMoSR's default; only DySample's end conv shows it

    # every conv (DOConv2d or ConvNXC) of the upsampler has an eval_conv
    convs = [i for i in range(32) if f"upsampler.{i}.eval_conv.weight" in state_dict]
    last_conv = state_dict[f"upsampler.{convs[-1]}.eval_conv.weight"] if convs else None
    dysample = next(
        (i for i in (0, 2) if f"upsampler.{i}.offset.weight" in state_dict), None
    )

    if scale == 1:
        # a single conv, whatever the upsampler
        assert last_conv is not None
        return upsampler, last_conv.shape[0], mid_dim, d_kernel
    if upsampler == "conv":
        # "conv" keeps the single conv of scale 1, so the output is
        # (H + 4 - 4 * scale) x (W + 4 - 4 * scale), not scale times the
        # input. With out_ch divisible by scale**2 its keys and shapes are
        # those of pixelshuffledirect with out_ch // scale**2.
        raise ValueError("SMoSR's 'conv' upsampler does not upscale (scale > 1)")
    if dysample is not None:
        # DySample, after a conv to mid_dim unless mid_dim is its input width
        end_conv = state_dict[f"upsampler.{dysample}.end_conv.weight"]
        mid_dim = state_dict[f"upsampler.{dysample}.offset.weight"].shape[1]
        return "dysample", end_conv.shape[0], mid_dim, end_conv.shape[2]
    assert last_conv is not None
    if 1 in convs:
        # nn.Upsample first, then a conv to mid_dim
        mid_dim = state_dict["upsampler.1.eval_conv.weight"].shape[0]
        return "pa_up", last_conv.shape[0], mid_dim, d_kernel
    if 2 in convs:
        # conv to mid_dim, LeakyReLU, conv + PixelShuffle steps
        mid_dim = state_dict["upsampler.0.eval_conv.weight"].shape[0]
        return "pixelshuffle", last_conv.shape[0], mid_dim, d_kernel
    if 3 in convs:
        # conv, nn.Upsample, LeakyReLU steps
        return "nearest+conv", last_conv.shape[0], mid_dim, d_kernel
    return "pixelshuffledirect", last_conv.shape[0] // scale**2, mid_dim, d_kernel


class SMoSRArch(Architecture[SMoSR]):
    def __init__(self) -> None:
        super().__init__(
            id="SMoSR",
            detect=KeyCondition.has_all(
                "short.weight",
                "short.bias",
                "blocks_1.0.body.0.eval_conv.weight",
                "blocks_1.0.body.4.eval_conv.weight",
                "blocks_1.1.body.0.eval_conv.weight",
                "blocks_1.1.body.4.eval_conv.weight",
                "end_block.0.body.0.eval_conv.weight",
                "end_block.0.body.4.eval_conv.weight",
                "end_block.1.eval_conv.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SMoSR]:
        in_ch = state_dict["short.weight"].shape[1]
        # short copies every input channel scale**2 times
        scale = math.isqrt(state_dict["short.weight"].shape[0] // in_ch)
        dim = state_dict["blocks_1.0.body.0.eval_conv.weight"].shape[0]
        # rep: ConvNXC (with a 1x1 sk branch) instead of DOConv2d
        rep = "blocks_1.0.body.0.sk.W" in state_dict
        n_mb = get_seq_len(state_dict, "blocks_2")
        upsampler, out_ch, upsampler_mid_dim, d_kernel = _get_upsampler(
            state_dict, scale
        )

        model = SMoSR(
            in_ch=in_ch,
            out_ch=out_ch,
            dim=dim,
            scale=scale,
            rep=rep,
            n_mb=n_mb,
            upsampler=upsampler,
            upsampler_mid_dim=upsampler_mid_dim,
            d_kernel=d_kernel,
        )

        tags = [f"{dim}dim", f"{n_mb}mb"]
        if scale != 1:
            tags.append(upsampler)
        if rep:
            tags.append("rep")

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
            output_channels=out_ch,
            # the model reflect-pads 2 pixels per side first
            size_requirements=SizeRequirements(minimum=3),
        )


__all__ = ["SMoSRArch", "SMoSR"]
