from __future__ import annotations

from pathlib import Path

import torch
from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    ModelDescriptor,
    StateDict,
)
from .__arch.smovsr_arch import FSMoSR, SMoVSR

# The frames after a kept frame that its output reads: the backward sweep reads
# the whole rest of a clip, and a window that starts it LOOK_AHEAD frames later
# stays within 1.4/255 of a whole-clip run (worst pixel, a generated moving clip).
LOOK_AHEAD = 16
# The frames swept forward, outputs discarded, before the first wanted frame of a
# cold start in the middle of a video.
WARM_UP = 8

# What a window carries to the next: the forward state after its last frame and
# the encodings and forward outputs of the frames after its kept ones.
Carry = tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]


def _sweep(
    sweep: FSMoSR, x: torch.Tensor, state: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of a sweep. Where `valid` is false, the step has no previous state:
    FSMoSR.step_wo_hystory and the model's clamp of the state it starts, bit for bit."""
    v = valid.view(-1, 1, 1, 1)
    h = torch.where(v, sweep.align(state), torch.zeros_like(state))
    if sweep.h_clamp is not None:
        h = h.clamp(-sweep.h_clamp, sweep.h_clamp)
    feat = sweep.blocks_2(x + h)
    out = sweep.end_block(feat + x)
    z = sweep.z(torch.cat([x, h], 1))
    if sweep.h_clamp is not None:
        feat = feat.clamp(-sweep.h_clamp, sweep.h_clamp)
    return out, torch.where(v, h + z * (feat - h), feat)


def encode_forward(
    model: SMoVSR, frame: torch.Tensor, state: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A B,3,H,W frame's encoding, forward output and the forward state after it, from
    the state after the frame before it (`valid` [B] bool false: none, the state's
    values unused)."""
    feat = model.fe(frame)
    out, state = _sweep(model.f_forward, feat, state, valid)
    return feat, out, state


def backward_step(
    model: SMoVSR,
    feat: torch.Tensor,
    forward_out: torch.Tensor,
    state: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A frame's backward output and backward state, from its encoding, its forward
    output and the backward state of the frame after it (`valid` false: none)."""
    x = model.backward_cat(torch.cat([feat, forward_out], 1))
    return _sweep(model.f_backward, x, state, valid)


def head(
    model: SMoVSR,
    frame: torch.Tensor,
    forward_out: torch.Tensor,
    backward_out: torch.Tensor,
) -> torch.Tensor:
    """A frame's B,3,sH,sW output in 0..1 from its two sweeps' outputs."""
    short = torch.nn.functional.interpolate(
        frame, scale_factor=model.scale, mode="bilinear", align_corners=False
    )
    out = model.upsampler(torch.cat([forward_out, backward_out], 1)) + short
    return out.clamp(0, 1)


def _dtype(model: SMoVSR) -> torch.dtype:
    return model.backward_cat.weight.dtype


def run_clip(model: SMoVSR, frames: torch.Tensor) -> torch.Tensor:
    """B,T,3,H,W frames in 0..1 to B,T,3,sH,sW in FP32 in 0..1: the model's sweeps
    over the whole clip. The frames are cast to the model's dtype."""
    return model(frames.to(_dtype(model))).clamp(0, 1).float()


def run_window(
    model: SMoVSR,
    frames: torch.Tensor,
    keep: int,
    past: Carry | None = None,
) -> tuple[torch.Tensor, Carry]:
    """The B,keep,3,sH,sW outputs in FP32 in 0..1 of a window of B,T,3,H,W frames and
    what to carry to the next window, from the carry `past` of the window before it
    or a cold start.

    The forward sweep is exact across windows: the first frames of a window are the
    frames after the previous window's kept ones, whose encodings and forward
    outputs come with `past`, and its state continues the previous window's last
    frame. The backward sweep starts at the window's last frame without a state;
    the window's last frame being the clip's, a window gives the whole clip's
    outputs (up to the rounding of a different batch)."""
    count = frames.shape[1]
    if not 1 <= keep <= count:
        raise ValueError(f"SMoVSR: keep must be in 1..{count}, got {keep}")
    clip = frames.to(_dtype(model)).unbind(1)
    batch, _, _, height, width = frames.shape
    if past is None:
        dim = model.backward_cat.out_channels
        state = clip[0].new_zeros(batch, dim, height, width)
        feats, outs = (), ()
    else:
        state, feats, outs = past
    warm = past is not None
    if len(feats) > count:
        raise ValueError(
            f"SMoVSR: the carry holds {len(feats)} encoded frames, the window only {count}"
        )
    feat_list = list(feats)
    out_list = list(outs)
    for t in range(len(feats), count):
        valid = torch.full((batch,), warm or t > 0, device=frames.device)
        feat, out, state = encode_forward(model, clip[t], state, valid)
        feat_list.append(feat)
        out_list.append(out)

    outputs: list[torch.Tensor] = [torch.empty(0)] * keep
    back = torch.zeros_like(state)
    for t in range(count - 1, -1, -1):
        valid = torch.full((batch,), t < count - 1, device=frames.device)
        back_out, back = backward_step(model, feat_list[t], out_list[t], back, valid)
        if t < keep:
            outputs[t] = head(model, clip[t], out_list[t], back_out).float()
    return torch.stack(outputs, dim=1), (
        state,
        tuple(feat_list[keep:]),
        tuple(out_list[keep:]),
    )


def _clamp_tag(h_clamp: float) -> str:
    return f"clamp{h_clamp:g}"


def apply_training_settings(
    descriptor: ModelDescriptor[torch.nn.Module], path: str | Path
) -> float | None:
    """Sets the state bound the SMoVSR file at `path` was trained with on its loaded
    model and returns it. A state dict alone cannot say it: neosr checkpoints (.pth)
    record it beside the weights (neosr -> network_g -> h_clamp). A file without that
    record (a .safetensors conversion, another trainer) keeps the model's bound,
    upstream's default of none, and this returns None."""
    model = descriptor.model
    if not isinstance(model, SMoVSR):
        raise TypeError(
            f"SMoVSR: training settings apply to an SMoVSR model, not {type(model).__name__}"
        )
    path = Path(path)
    if path.suffix.lower() not in (".pth", ".pt", ".ckpt"):
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    options = checkpoint.get("neosr") if isinstance(checkpoint, dict) else None
    network = options.get("network_g") if isinstance(options, dict) else None
    if not isinstance(network, dict) or "h_clamp" not in network:
        return None
    value = network["h_clamp"]
    if value is None:
        h_clamp = None
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"SMoVSR: the training setting h_clamp of {path.name} must be a number or None, got {value!r}"
        )
    elif not value > 0:
        raise ValueError(
            f"SMoVSR: the training setting h_clamp of {path.name} must be positive, got {value!r}"
        )
    else:
        h_clamp = float(value)

    model.f_forward.h_clamp = h_clamp
    model.f_backward.h_clamp = h_clamp
    model.hyperparameters["h_clamp"] = h_clamp
    descriptor.tags[:] = [t for t in descriptor.tags if not t.startswith("clamp")]
    if h_clamp is not None:
        descriptor.tags.append(_clamp_tag(h_clamp))
    return h_clamp


class SMoVSRArch(Architecture[SMoVSR]):
    def __init__(self) -> None:
        super().__init__(
            id="SMoVSR",
            detect=KeyCondition.has_all(
                "fe.0.body.0.eval_conv.weight",
                "fe.1.gate_conv.0.weight",
                "f_forward.blocks_2.0.body.0.eval_conv.weight",
                "f_forward.end_block.1.eval_conv.weight",
                "f_forward.z.0.weight",
                "f_backward.z.0.weight",
                "backward_cat.weight",
                "upsampler.0.weight",
                "upsampler.1.weight",
                "upsampler.2.MetaUpsample",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[SMoVSR]:
        in_ch = state_dict["fe.0.body.0.eval_conv.weight"].shape[1]
        dim = state_dict["fe.0.body.0.eval_conv.weight"].shape[0]
        # rep: ConvNXC (with a 1x1 sk branch) instead of DOConv2d
        rep = "fe.0.body.0.sk.W" in state_dict
        # UniUpsampleV4_light records [version, upsampler, scale, ...] in a buffer
        scale = int(state_dict["upsampler.2.MetaUpsample"][2])
        align = "f_forward.align.eval_conv.weight"
        # a depthwise conv, or InceptionDWConv2d's branches when align_kernel is None
        align_kernel = state_dict[align].shape[2] if align in state_dict else None

        model = SMoVSR(
            in_ch=in_ch,
            dim=dim,
            rep=rep,
            scale=scale,
            align_kernel=align_kernel,
        )
        # Strict loading; the stored merged convolutions are then made again from
        # the training-form weights by eval() (a file's stored ones can be stale),
        # and the descriptor, which loads its state dict again, gets the model's.
        model.load_state_dict(state_dict)
        model.eval()

        tags = [f"{dim}dim", "temporal"]
        if rep:
            tags.append("rep")

        return ImageModelDescriptor(
            model,
            model.state_dict(),
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=False,
            scale=scale,
            input_channels=in_ch,
            output_channels=3,
        )


__all__ = [
    "LOOK_AHEAD",
    "WARM_UP",
    "Carry",
    "SMoVSR",
    "SMoVSRArch",
    "apply_training_settings",
    "backward_step",
    "encode_forward",
    "head",
    "run_clip",
    "run_window",
]
