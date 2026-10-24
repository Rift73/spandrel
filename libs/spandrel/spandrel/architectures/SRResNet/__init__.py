from __future__ import annotations

import math
import re
from typing import Literal, Union

from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from ...util import KeyCondition, get_seq_len
from .__arch.srresnet import MSRResNet, SRResNet, SRResNetSequential

_SRModel = Union[SRResNet, SRResNetSequential, MSRResNet]

_upscale_module_re = re.compile(r"^(upscale\d+)x\.0\.weight$")
_seq_conv_re = re.compile(r"^model\.(\d+)\.weight$")


class SRResNetArch(Architecture[_SRModel]):
    def __init__(self) -> None:
        super().__init__(
            id="SRResNet",
            detect=KeyCondition.has_any(
                # layout 1: delldu/SRResNet and variants
                KeyCondition.has_all(
                    "conv_input.weight",
                    "conv_mid.weight",
                    "bn_mid.weight",
                    "conv_output.weight",
                    "residual.0.conv1.weight",
                    KeyCondition.has_any(
                        "residual.0.in1.weight",
                        "residual.0.bn1.weight",
                    ),
                    KeyCondition.has_any(
                        "upscale2x.0.weight",
                        "upscale4x.0.weight",
                        "upscale8x.0.weight",
                    ),
                ),
                # layout 2: "Sequential" layout (e.g. 2x-Waifaux-NL3-SRResNet)
                # NB: disjoint from ESRGAN, whose blocks are
                # `model.1.sub.N.RDB1.conv1...` instead of `model.1.sub.N.res.0...`
                KeyCondition.has_all(
                    "model.0.weight",
                    "model.1.sub.0.res.0.weight",
                    "model.2.weight",
                    "model.5.weight",
                ),
                # layout 3: BasicSR "Modified SRResNet" (MSRResNet)
                KeyCondition.has_all(
                    "conv_first.weight",
                    "body.0.conv1.weight",
                    "upconv1.weight",
                    "conv_hr.weight",
                    "conv_last.weight",
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[_SRModel]:
        if "model.0.weight" in state_dict and "model.1.sub.0.res.0.weight" in state_dict:
            return self._load_sequential(state_dict)
        if "conv_first.weight" in state_dict and "body.0.conv1.weight" in state_dict:
            return self._load_msrresnet(state_dict)
        return self._load_delldu(state_dict)

    # ------------------------------------------------------------------
    # layout 1: delldu/SRResNet and variants
    # ------------------------------------------------------------------

    def _load_delldu(self, state_dict: StateDict) -> ImageModelDescriptor[SRResNet]:
        in_nc = state_dict["conv_input.weight"].shape[1]
        out_nc = state_dict["conv_output.weight"].shape[0]
        nf = state_dict["conv_input.weight"].shape[0]
        nb = get_seq_len(state_dict, "residual")
        kernel_size = state_dict["conv_input.weight"].shape[-1]

        # norm flavor: `in1`/`in2` (InstanceNorm, original) or `bn1`/`bn2`
        # (BatchNorm variants)
        norm: Literal["instance", "batch"] = "instance"
        stats_key = "in1"
        if "residual.0.bn1.weight" in state_dict:
            norm = "batch"
            stats_key = "bn1"

        # old checkpoints (PyTorch <= 0.4) always store running buffers,
        # even for InstanceNorm; newer ones don't
        with_running_stats = f"residual.0.{stats_key}.running_mean" in state_dict

        # find the upsampler module (`upscale2x`/`upscale4x`/`upscale8x`)
        upsampler_name = "upscale4x"
        for key in state_dict:
            match = _upscale_module_re.match(key)
            if match:
                upsampler_name = match.group(1) + "x"
                break

        # each upsampler stage is [conv, PixelShuffle(2), LeakyReLU], so the
        # first conv maps nf -> nf * 4 and the total scale is 2^num_stages
        num_stages = sum(
            1 for i in range(10) if f"{upsampler_name}.{i}.weight" in state_dict
        )
        up_channels = state_dict[f"{upsampler_name}.0.weight"].shape[0]
        stage_up = up_channels // nf
        if up_channels % nf != 0 or math.isqrt(stage_up) ** 2 != stage_up:
            raise ValueError(
                f"Unexpected upsampler conv shape {up_channels}x{nf} "
                f"for {upsampler_name}"
            )
        scale = math.isqrt(stage_up) ** num_stages

        bias = "conv_input.bias" in state_dict

        model = SRResNet(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            nb=nb,
            upscale=scale,
            kernel_size=kernel_size,
            norm=norm,
            with_running_stats=with_running_stats,
            bias=bias,
        )

        # make the state dict strictly loadable:
        # - old checkpoints miss `num_batches_tracked` buffers (introduced in
        #   newer PyTorch versions); add zero-initialized ones
        # - drop norm buffers that the model does not register (e.g. running
        #   stats of checkpoints of models without `track_running_stats`)
        model_state = model.state_dict()
        for key, value in model_state.items():
            if key.endswith("num_batches_tracked") and key not in state_dict:
                state_dict[key] = value
        for key in list(state_dict.keys()):
            if key not in model_state and re.search(
                r"(running_mean|running_var|num_batches_tracked)$", key
            ):
                del state_dict[key]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{nf}nf", f"{nb}nb"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_nc,
            output_channels=out_nc,
        )

    # ------------------------------------------------------------------
    # layout 2: "Sequential" layout (e.g. 2x-Waifaux-NL3-SRResNet)
    # ------------------------------------------------------------------

    def _load_sequential(
        self, state_dict: StateDict
    ) -> ImageModelDescriptor[SRResNetSequential]:
        conv_indices = sorted(
            int(match.group(1))
            for key in state_dict
            if (match := _seq_conv_re.match(key))
        )
        # `model.1` is the trunk wrapper; it holds no direct weights
        if len(conv_indices) < 4 or conv_indices[0] != 0 or 1 in conv_indices:
            raise ValueError(f"Unexpected conv layout at {conv_indices}")

        *stage_conv_indices, hr_index, last_index = conv_indices[1:]

        in_nc = state_dict["model.0.weight"].shape[1]
        nf = state_dict["model.0.weight"].shape[0]
        out_nc = state_dict[f"model.{last_index}.weight"].shape[0]
        bias = "model.0.bias" in state_dict

        # trunk: `nb` residual blocks followed by a trunk conv
        nb = get_seq_len(state_dict, "model.1.sub") - 1
        if nb < 1 or f"model.1.sub.{nb}.weight" not in state_dict:
            raise ValueError("Expected a trunk conv at the end of model.1.sub")

        # upsampler: single stage [conv(nf -> nf*scale^2), PixelShuffle] for
        # 2x/3x, or `k` stages of [conv(nf -> nf*4), PixelShuffle(2)] for 4x/8x
        if len(stage_conv_indices) == 1:
            up_channels = state_dict[f"model.{stage_conv_indices[0]}.weight"].shape[0]
            ratio = up_channels // nf
            scale = math.isqrt(ratio)
            if (
                up_channels % nf != 0
                or scale * scale != ratio
                or scale not in (2, 3)
            ):
                raise ValueError(
                    f"Unexpected upsampler conv shape {up_channels}x{nf}"
                )
        else:
            for index in stage_conv_indices:
                if (
                    state_dict[f"model.{index}.weight"].shape[0]
                    != nf * 4
                ):
                    raise ValueError(
                        f"Unexpected upsampler conv shape at model.{index}"
                    )
            scale = 2 ** len(stage_conv_indices)

        # the conv before the output must keep the feature width
        if state_dict[f"model.{hr_index}.weight"].shape[0] != nf:
            raise ValueError(f"Unexpected conv shape at model.{hr_index}")

        model = SRResNetSequential(
            in_nc=in_nc,
            out_nc=out_nc,
            nf=nf,
            nb=nb,
            scale=scale,
            bias=bias,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR",
            tags=[f"{nf}nf", f"{nb}nb"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_nc,
            output_channels=out_nc,
        )

    # ------------------------------------------------------------------
    # layout 3: BasicSR "Modified SRResNet" (MSRResNet)
    # ------------------------------------------------------------------

    def _load_msrresnet(
        self, state_dict: StateDict
    ) -> ImageModelDescriptor[MSRResNet]:
        num_feat = state_dict["conv_first.weight"].shape[0]
        num_in_ch = state_dict["conv_first.weight"].shape[1]
        num_out_ch = state_dict["conv_last.weight"].shape[0]
        num_block = get_seq_len(state_dict, "body")

        # 4x models have a second upconv; 2x/3x have a single one whose
        # output width is num_feat * upscale^2
        if "upconv2.weight" in state_dict:
            upscale = 4
            if state_dict["upconv1.weight"].shape[0] != num_feat * 4:
                raise ValueError("Unexpected upconv1 shape for a 4x model")
        else:
            ratio = state_dict["upconv1.weight"].shape[0] // num_feat
            upscale = math.isqrt(ratio)
            if (
                state_dict["upconv1.weight"].shape[0] % num_feat != 0
                or upscale * upscale != ratio
                or upscale not in (2, 3)
            ):
                raise ValueError("Unexpected upconv1 shape")

        model = MSRResNet(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat,
            num_block=num_block,
            upscale=upscale,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR",
            tags=[f"{num_feat}nf", f"{num_block}nb"],
            supports_half=True,
            supports_bfloat16=True,
            scale=upscale,
            input_channels=num_in_ch,
            output_channels=num_out_ch,
        )


__all__ = ["SRResNetArch", "SRResNet", "SRResNetSequential", "MSRResNet"]
