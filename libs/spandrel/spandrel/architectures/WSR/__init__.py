from __future__ import annotations

import math
from typing import Union

from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.wsr_arch import WESR, WideContextSR

_WSR = Union[WESR, WideContextSR]

# (width, hidden, depth, gates, context_width, heads, attention) -> preset name
_WESR_PRESETS = {
    (64, 128, 7, 3, 64, 2, True): "Default",
    (32, 64, 9, 3, 64, 1, True): "Compact",
}
# (width, hidden, depth, sft_count, context_width) -> preset name
_CONTEXT_PRESETS = {
    (32, 64, 5, 2, 64): "UltraCompact",
    (16, 32, 5, 2, 32): "SuperUltraCompact",
    (8, 16, 2, 1, 32): "Mobile",
}


def _conv3xc(prefix: str) -> KeyCondition:
    # A Conv3XC in training form, or folded into one 3x3 conv.
    return KeyCondition.has_any(f"{prefix}.sk.weight", f"{prefix}.weight")


def _rgb5(prefix: str) -> KeyCondition:
    # A FoldableRGB5 in training form, or folded into one 5x5 conv.
    return KeyCondition.has_any(f"{prefix}.first.sk.weight", f"{prefix}.weight")


def _channels(state_dict: StateDict, prefix: str) -> tuple[int, int]:
    """The output and input channels of a Conv3XC in either form."""
    weight = state_dict.get(f"{prefix}.sk.weight")
    if weight is None:
        weight = state_dict[f"{prefix}.weight"]
    return weight.shape[0], weight.shape[1]


class WSRArch(Architecture[_WSR]):
    def __init__(self) -> None:
        super().__init__(
            id="WSR",
            detect=KeyCondition.has_all(
                _conv3xc("stem"),
                _conv3xc("blocks.0.expand"),
                _conv3xc("blocks.0.project"),
                _conv3xc("context_stem"),
                _conv3xc("context_blocks.0.expand"),
                _conv3xc("context_blocks.3.project"),
                "merge.weight",
                "merge.bias",
                "gates.0.project.weight",
                "gates.0.local_scale",
                "reconstruct.weight",
                KeyCondition.has_any(
                    # WESR
                    KeyCondition.has_all(
                        _rgb5("first_expand"),
                        _conv3xc("first_project"),
                        _rgb5("base"),
                    ),
                    # WideContextSR
                    _conv3xc("deep"),
                ),
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[_WSR]:
        # Folded (exported) checkpoints hold one plain conv per Conv3XC and
        # FoldableRGB5.
        folded = "stem.weight" in state_dict

        context_width, width = state_dict["merge.weight"].shape[:2]
        hidden = _channels(state_dict, "blocks.0.expand")[0]
        blocks = get_seq_len(state_dict, "blocks")
        gates = get_seq_len(state_dict, "gates")
        scale = math.isqrt(state_dict["reconstruct.weight"].shape[0] // 3)

        model: _WSR
        if _conv3xc("first_project")(state_dict):
            # first_expand/first_project form WESR's first block
            depth = blocks + 1
            use_xsa = "attention.logit_scale" in state_dict
            if use_xsa:
                heads = state_dict["attention.logit_scale"].shape[0]
            else:
                # Without attention, heads leave no trace: the default 2 where
                # the width allows it.
                heads = 2 if width % 16 == 0 else 1
            model = WESR(
                depth=depth,
                gates=gates,
                # cannot be deduced from the state dict (32 or 64)
                window=64,
                use_xsa=use_xsa,
                width=width,
                hidden=hidden,
                context_width=context_width,
                heads=heads,
                scale=scale,
            )
            preset = _WESR_PRESETS.get(
                (width, hidden, depth, gates, context_width, heads, use_xsa)
            )
        else:
            depth = blocks
            model = WideContextSR(
                width=width,
                hidden=hidden,
                depth=depth,
                sft_count=gates,
                context_width=context_width,
                scale=scale,
            )
            preset = _CONTEXT_PRESETS.get((width, hidden, depth, gates, context_width))

        tags = [f"{width}nf", f"{depth}nb"]
        if preset is not None:
            tags.insert(0, preset)
        if isinstance(model, WESR) and model.attention is None:
            tags.append("no-XSA")

        if folded:
            # The strict load below needs the folded modules.
            model.eval().prepare_for_export()

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
        )

        # Fold the training branches for inference after the strict load above,
        # on CPU in FP32 (the factors in FP64) before ModelLoader moves this
        # module to the GPU and chaiNNer optionally calls half().
        model.eval()
        if not folded:
            model.prepare_for_export()
        elif isinstance(model, WESR) and model.attention is not None:
            # The attention's deployment scale is a non-persistent buffer: set
            # it from the logit scale loaded above.
            model.attention.prepare_for_export()
        return descriptor


__all__ = ["WSRArch", "WESR", "WideContextSR"]
