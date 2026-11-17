from __future__ import annotations

import math

import torch
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from ..WSR import _channels, _conv3xc, _rgb5
from .__arch.rsr_arch import RSR

# What a window carries to the next: the forward state of its last kept frame
# and the encodings of the frames after it.
_Carry = tuple[
    torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
]


def run_clip(model: RSR, frames: torch.Tensor) -> torch.Tensor:
    """B,T,3,H,W frames in 0..1 to B,T,3,sH,sW in FP32: the model's sweeps over the
    whole clip. The frames are cast to the model's dtype."""
    return model(frames.to(model.reconstruct.weight.dtype)).float()


def run_window(
    model: RSR,
    frames: torch.Tensor,
    keep: int,
    past: _Carry | None = None,
) -> tuple[torch.Tensor, _Carry]:
    """The B,keep,3,sH,sW outputs in FP32 of a window of B,T,3,H,W frames and what to
    carry to the next window, from the carry `past` of the window before it or a cold
    start.

    Every frame is encoded once: the first frames of a window are the frames after
    the previous window's kept ones, and their encodings come with `past`. The
    reverse sweep starts at the window's last frame, which has no future; the
    forward sweep reads `past`'s state at the first frame. A window covering a whole
    clip with keep = T equals run_clip(), and chained windows equal
    RSR.infer(chunk=keep, lookahead=T - keep)."""
    count = frames.shape[1]
    if not 1 <= keep <= count:
        raise ValueError(f"RSR: keep must be in 1..{count}, got {keep}")
    clip = list(frames.to(model.reconstruct.weight.dtype).unbind(1))
    state, carried = (None, ()) if past is None else past
    if len(carried) > count:
        raise ValueError(
            f"RSR: the carry holds {len(carried)} encoded frames, the window only {count}"
        )
    encoded = [*carried, *(model.encode(frame) for frame in clip[len(carried) :])]

    reverse: list[torch.Tensor | None] = [None] * (count + 1)
    for t in range(count - 1, 0, -1):
        _, reverse[t] = model.step_encoded(
            clip[t], encoded[t], future=reverse[t + 1], output=False
        )
    outputs: list[torch.Tensor] = []
    for t in range(keep):
        _, state = model.step_encoded(
            clip[t], encoded[t], state, reverse[t + 1], output=False
        )
        outputs.append(model.head(clip[t], state).float())
    # keep >= 1, so the forward sweep set the state
    assert state is not None
    return torch.stack(outputs, dim=1), (state, tuple(encoded[keep:]))


class RSRArch(Architecture[RSR]):
    def __init__(self) -> None:
        super().__init__(
            id="RSR",
            detect=KeyCondition.has_all(
                # WSR's attention backbone (WESR)
                _conv3xc("stem"),
                _rgb5("first_expand"),
                _conv3xc("first_project"),
                _conv3xc("blocks.0.expand"),
                _conv3xc("blocks.0.project"),
                _conv3xc("context_stem"),
                _conv3xc("context_blocks.0.expand"),
                _conv3xc("context_blocks.3.project"),
                "merge.weight",
                "merge.bias",
                "gates.0.project.weight",
                "gates.0.local_scale",
                "attention.logit_scale",
                "reconstruct.weight",
                _rgb5("base"),
                # the temporal read of the neighbouring frames' states
                "attention.key_value_past.weight",
                "attention.key_value_future.weight",
                "attention.null_key",
                "attention.temporal_gain",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[RSR]:
        # Folded (exported) checkpoints hold one plain conv per Conv3XC and
        # FoldableRGB5.
        folded = "stem.weight" in state_dict

        context_width, width = state_dict["merge.weight"].shape[:2]
        hidden = _channels(state_dict, "blocks.0.expand")[0]
        # first_expand/first_project form the first block
        depth = get_seq_len(state_dict, "blocks") + 1
        gates = get_seq_len(state_dict, "gates")
        heads = state_dict["attention.logit_scale"].shape[0]
        scale = math.isqrt(state_dict["reconstruct.weight"].shape[0] // 3)

        model = RSR(
            depth=depth,
            gates=gates,
            width=width,
            hidden=hidden,
            context_width=context_width,
            heads=heads,
            scale=scale,
        )

        tags = [f"{width}nf", f"{depth}nb", "temporal"]
        if (width, hidden, depth, gates, context_width, heads) == (32, 64, 9, 3, 64, 1):
            tags.insert(0, "Compact")

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
        # module to the GPU and chaiNNer optionally calls half(). One image is
        # one frame without neighbours.
        model.eval()
        if not folded:
            model.prepare_for_export()
        else:
            # The attention's deployment scale is a non-persistent buffer: set
            # it from the logit scale loaded above.
            model.attention.prepare_for_export()
        return descriptor


__all__ = ["RSRArch", "RSR", "run_clip", "run_window"]
