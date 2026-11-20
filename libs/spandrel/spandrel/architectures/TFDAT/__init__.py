from __future__ import annotations

import math
from typing import TypedDict, TypeVar

import torch
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from ..FDAT.__arch.fdat import SampleMods3
from .__arch.tfdat import TFDAT
from .__arch.tfdat_v2 import TFDATv2

_DEFAULT_PATTERN = ["spatial", "channel"]

M = TypeVar("M", TFDAT, TFDATv2)


class _Settings(TypedDict):
    num_in_ch: int
    num_out_ch: int
    scale: int
    embed_dim: int
    num_groups: int
    depth_per_group: int
    num_heads: int
    window_size: int
    ffn_expansion_ratio: float
    aim_reduction_ratio: int
    group_block_pattern: list[str] | None
    mid_dim: int
    upsampler_type: SampleMods3


def _call(model: TFDAT | TFDATv2, image: torch.Tensor) -> torch.Tensor:
    """One image is a still clip: the frame repeated clip_size times."""
    return model(image[:, None].repeat(1, model.clip_size, 1, 1, 1))


def _settings(state_dict: StateDict, conv_first: str, num_in_ch: int) -> _Settings:
    """The settings TFDAT and TFDATv2 share, from their FDAT backbone, their first
    convolution (`conv_first`, weight key) and their input channels."""
    _, upsampler_index, upsampler_scale, _, num_out_ch, mid_dim, _ = state_dict[
        "upsampler.MetaUpsample"
    ].tolist()
    embed_dim, conv_in_ch = state_dict[conv_first].shape[:2]
    # Scales 1 and 2 unshuffle by 4 and 2 into an internal scale of 4.
    unshuffle = math.isqrt(conv_in_ch // num_in_ch)
    scale = upsampler_scale if unshuffle == 1 else 4 // unshuffle

    # Spatial blocks hold the window attention's bias, channel blocks a temperature.
    types = [
        "spatial" if f"groups.0.blocks.{i}.attn.bias" in state_dict else "channel"
        for i in range(get_seq_len(state_dict, "groups.0.blocks"))
    ]
    period = next(
        p
        for p in range(1, len(types) + 1)
        if len(types) % p == 0 and types == types[:p] * (len(types) // p)
    )
    pattern = types[:period]
    if "spatial" in types:
        bias = state_dict[f"groups.0.blocks.{types.index('spatial')}.attn.bias"]
        num_heads = bias.shape[0]
        window_size = math.isqrt(bias.shape[1])
    else:
        # the window size is only in the spatial blocks' bias
        num_heads = state_dict["groups.0.blocks.0.attn.temp"].shape[0]
        window_size = 8

    return {
        "num_in_ch": num_in_ch,
        "num_out_ch": num_out_ch,
        "scale": scale,
        "embed_dim": embed_dim,
        "num_groups": get_seq_len(state_dict, "groups"),
        "depth_per_group": len(types) // period,
        "num_heads": num_heads,
        "window_size": window_size,
        "ffn_expansion_ratio": float(
            state_dict["groups.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim
        ),
        "aim_reduction_ratio": embed_dim
        // state_dict["groups.0.blocks.0.inter.cg.1.weight"].shape[0],
        "group_block_pattern": None if pattern == _DEFAULT_PATTERN else pattern,
        "mid_dim": mid_dim,
        "upsampler_type": list(SampleMods3.__args__)[upsampler_index],
    }


def _descriptor(
    arch: Architecture[M],
    model: M,
    state_dict: StateDict,
    settings: _Settings,
) -> ImageModelDescriptor[M]:
    scale = model.upscale
    return ImageModelDescriptor(
        model,
        state_dict,
        architecture=arch,
        purpose="Restoration" if scale == 1 else "SR",
        tags=[
            f"{settings['embed_dim']}dim",
            settings["upsampler_type"],
            f"clip {model.clip_size}",
            "temporal",
        ],
        supports_half=True,
        supports_bfloat16=True,
        scale=scale,
        input_channels=model.num_in_ch,
        output_channels=settings["num_out_ch"],
        call_fn=_call,
    )


class TFDATArch(Architecture[TFDAT]):
    def __init__(self) -> None:
        super().__init__(
            id="TFDAT",
            detect=KeyCondition.has_all(
                "flow_net.enc.0.weight",
                "flow_net.flow_pred.weight",
                "temporal_fuse.0.weight",
                "temporal_fuse.2.weight",
                "conv_first.weight",
                "groups.0.blocks.0.n1.weight",
                "groups.0.blocks.0.ffn.fc1.weight",
                "groups.0.blocks.0.inter.cg.1.weight",
                "conv_after.weight",
                "upsampler.MetaUpsample",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[TFDAT]:
        flow_first = state_dict["flow_net.enc.0.weight"]
        settings = _settings(state_dict, "conv_first.weight", flow_first.shape[1] // 2)
        # The clip size is in no weight (the neighbours are averaged), and the
        # trainer always builds TFDAT with 5 frames.
        model = TFDAT(**settings, clip_size=5, flow_base_ch=flow_first.shape[0])
        return _descriptor(self, model, state_dict, settings)


class TFDATv2Arch(Architecture[TFDATv2]):
    def __init__(self) -> None:
        super().__init__(
            id="TFDATv2",
            detect=KeyCondition.has_all(
                "flow_net.enc1.0.weight",
                "flow_net.enc3.2.weight",
                "flow_net.dec3.4.weight",
                "flow_net.dec1.4.weight",
                "conv_first.0.weight",
                "conv_first.2.weight",
                "align_refine.body.0.weight",
                "temporal_fusion.fuse.0.weight",
                "temporal_fusion.conf.2.weight",
                "groups.0.blocks.0.n1.weight",
                "groups.0.blocks.0.ffn.fc1.weight",
                "conv_after.weight",
                "hr_refine.2.weight",
                "upsampler.MetaUpsample",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[TFDATv2]:
        flow_first = state_dict["flow_net.enc1.0.weight"]
        settings = _settings(state_dict, "conv_first.0.weight", flow_first.shape[1])
        embed_dim = state_dict["conv_first.0.weight"].shape[0]
        clip_size = state_dict["temporal_fusion.fuse.0.weight"].shape[1] // embed_dim
        model = TFDATv2(
            **settings, clip_size=clip_size, flow_base_ch=flow_first.shape[0]
        )
        return _descriptor(self, model, state_dict, settings)


__all__ = ["TFDAT", "TFDATArch", "TFDATv2", "TFDATv2Arch"]
