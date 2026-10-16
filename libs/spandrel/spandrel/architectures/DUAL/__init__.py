from __future__ import annotations

import math
from itertools import combinations

import torch
import torch.nn as nn
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from .__arch.dual_arch import (
    DUAL,
    PremixedDeployedDynamicConv3x3,
    TokenNorm,
    _premixed_conv3x3,
)


class _FP16SafeDynamicConv(PremixedDeployedDynamicConv3x3):
    """The last group's dynamic convolution, in FP32 when the model runs in FP16.

    Its output can far exceed FP16's 65504 (125k in a trained 1x Light) although
    the final norm right after it scales it back down. Only this convolution and
    the group's residual sum run in FP32; the persistent weights stay FP16.
    """

    def __init__(self, source: PremixedDeployedDynamicConv3x3) -> None:
        nn.Module.__init__(self)
        # Reuse loaded parameters under exactly the same state-dict keys.
        self.base = source.base
        self.experts = source.experts
        self.router = source.router

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or x.dtype != torch.float16:
            return super().forward(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return _premixed_conv3x3(
                x.float(),
                self.base.weight.float(),
                self.base.bias.float(),
                self.experts.float(),
                self.coefficients(x),
            )


class _FP16SafeTokenNorm(TokenNorm):
    """A norm reading the FP32 sum above; it hands the trunk back its own dtype."""

    def __init__(self, source: TokenNorm) -> None:
        nn.Module.__init__(self)
        self.norm = source.norm

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).to(self.norm.weight.dtype)


# Original layer indices of a full group: 2 and 5 are the context layers.
_FULL = (0, 1, 2, 3, 4, 5)
_CONTEXT = (2, 5)
# The schedule of the dual_light and dual_xs presets.
_LIGHT_INDICES = ((0, 1, 2, 3), (3, 4, 5), (0, 1, 2), (0, 1, 2, 3, 4, 5))
# (embed_dim, depths, pixel_unshuffle) -> preset name
_PRESETS = {
    (128, (4, 3, 3, 6), True): "Light",
    (128, (4, 3, 3, 6), False): "XS",
    (96, (6,) * 4, False): "DUAL3X",
    (160, (6,) * 6, False): "S",
    (192, (6,) * 6, False): "M",
    (224, (6,) * 10, False): "L",
    (256, (6,) * 12, False): "XL",
}


def _layer_indices(
    state_dict: StateDict, depths: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """The original index of every kept layer. Only the position of the context
    layers is in the weights: a pruned schedule they do not determine must be
    the Light/XS preset's."""
    options: list[list[tuple[int, ...]]] = []
    for group, depth in enumerate(depths):
        context = tuple(
            f"groups.{group}.layers.{i}.dual.theta" in state_dict for i in range(depth)
        )
        options.append(
            [
                indices
                for indices in combinations(_FULL, depth)
                if tuple(i in _CONTEXT for i in indices) == context
                and (group > 0 or indices[:2] == (0, 1))
            ]
        )
    if len(depths) == len(_LIGHT_INDICES) and all(
        preset in group for preset, group in zip(_LIGHT_INDICES, options)
    ):
        return _LIGHT_INDICES
    if all(len(group) == 1 for group in options):
        return tuple(group[0] for group in options)
    raise ValueError(
        "Unsupported DUAL layer schedule: the kept layers of a pruned group are"
        " not determined by the weights"
    )


class DUALArch(Architecture[DUAL]):
    def __init__(self) -> None:
        super().__init__(
            id="DUAL",
            detect=KeyCondition.has_all(
                "mean",
                KeyCondition.has_any("stem.weight", "stem.direct.expand.weight"),
                "groups.0.layers.0.attention.relative_bias",
                "groups.0.layers.0.attention.qkv.weight",
                "groups.0.layers.0.ffn.depthwise.weight",
                "groups.0.conv.experts",
                "groups.0.conv.router.fc1.weight",
                "groups.1.conv.experts",
                "final_norm.norm.weight",
                "xr.log_tau",
                "xr.query.weight",
                "xr.key.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[DUAL]:
        # Folded (exported) checkpoints hold one plain 3x3 conv per EDBB conv.
        folded = "stem.weight" in state_dict
        if folded:
            in_channels = state_dict["stem.weight"].shape[1]
            embed_dim = state_dict["stem.weight"].shape[0]
        else:
            in_channels = state_dict["stem.direct.expand.weight"].shape[1]
            embed_dim = state_dict["stem.direct.reduce.weight"].shape[0]
        pixel_unshuffle = in_channels == 12

        # tail: first conv, one conv per x2 stage (+1 for PU2), last conv
        tail_convs = (get_seq_len(state_dict, "tail") + 1) // 2
        scale = 2 ** (tail_convs - 2 - int(pixel_unshuffle))

        depths = tuple(
            get_seq_len(state_dict, f"groups.{group}.layers")
            for group in range(get_seq_len(state_dict, "groups"))
        )
        indices = _layer_indices(state_dict, depths)

        table = state_dict["groups.0.layers.0.attention.relative_bias"].shape[0]
        g_window = (math.isqrt(table) + 1) // 2

        c_pos_bias = "rfb"
        rfb_rank = 16
        context = next(
            (
                key[: -len("dual.theta")]
                for key in state_dict
                if key.endswith(".dual.theta")
            ),
            None,
        )
        if context is not None:
            bias = context + "attention.factored_bias."
            if bias + "frequency" in state_dict:
                rfb_rank = state_dict[bias + "frequency"].shape[0] * 2
            elif bias + "to_query" in state_dict:
                c_pos_bias = "rib"
                rfb_rank = state_dict[bias + "to_query"].shape[2]
            else:
                c_pos_bias = "none"

        qk = (
            "cosine"
            if "groups.0.layers.0.attention.logit_scale" in state_dict
            else "dot"
        )
        dynamic_experts = state_dict["groups.0.conv.experts"].shape[0]

        model = DUAL(
            scale=scale,
            embed_dim=embed_dim,
            depths=depths,
            # the factories leave a full schedule implicit
            layer_indices=None if all(i == _FULL for i in indices) else indices,
            pixel_unshuffle=pixel_unshuffle,
            dynamic_experts=dynamic_experts,
            # c_windows cannot be deduced from the state dict (default 32/64)
            g_window=g_window,
            c_pos_bias=c_pos_bias,
            rfb_rank=rfb_rank,
            qk=qk,
        )
        if folded:
            model.eval()
            model.prepare_for_export()

        tags = [f"{embed_dim}dim", f"{len(depths)}g"]
        preset = _PRESETS.get((embed_dim, depths, pixel_unshuffle))
        if preset is not None:
            tags.insert(0, preset)
        if pixel_unshuffle:
            tags.append("PU2")

        descriptor = ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=3,
            output_channels=3,
            size_requirements=SizeRequirements(minimum=4),
        )

        # Fold the static EDBB bases for inference after the strict load above;
        # the routers and expert kernels stay input-dependent. Fold on CPU in
        # FP32 (the factors in FP64) before ModelLoader moves this module to the
        # GPU and chaiNNer optionally calls half().
        model.float().eval()
        if not folded:
            model.prepare_for_export()
        last = model.groups[-1]
        assert isinstance(last.conv, PremixedDeployedDynamicConv3x3)
        last.conv = _FP16SafeDynamicConv(last.conv)
        model.final_norm = _FP16SafeTokenNorm(model.final_norm)
        if len(model.groups) == 2:
            # The cross-scale retrieval reads group 1's output too.
            model.xr.norm_u = _FP16SafeTokenNorm(model.xr.norm_u)
        model.requires_grad_(False).eval()
        return descriptor


__all__ = ["DUALArch", "DUAL"]
