from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import save_file

from spandrel import MAIN_REGISTRY, ImageModelDescriptor, ModelLoader
from spandrel.architectures.SMoSR import SMoSR
from spandrel.architectures.SMoVSR import (
    LOOK_AHEAD,
    SMoVSR,
    SMoVSRArch,
    apply_training_settings,
    backward_step,
    encode_forward,
    head,
    run_clip,
    run_window,
)

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)

# The released file (2x, dim 32, rep, align_kernel 7, trained with h_clamp 30); not
# in the repository: its tests run when SPANDREL_TEST_SMOVSR_MODEL holds its full path.
REAL_FILE = Path(os.environ.get("SPANDREL_TEST_SMOVSR_MODEL") or "")

VARIANTS = {
    "default": lambda: SMoVSR(),
    "plain": lambda: SMoVSR(rep=False),
    "x1": lambda: SMoVSR(scale=1),
    "x3": lambda: SMoVSR(scale=3),
    "x4": lambda: SMoVSR(scale=4, rep=False),
    "dim16": lambda: SMoVSR(dim=16),
    "align5": lambda: SMoVSR(align_kernel=5),
    "inception": lambda: SMoVSR(align_kernel=None),
    "clamped": lambda: SMoVSR(h_clamp=2.0),
}


def _build(name: str) -> SMoVSR:
    """Every parameter moved off its initial value (zero-initialized layers, such as
    the gates and the DOConv2d depth weights, would hide errors), in eval mode."""
    torch.manual_seed(0)
    model = VARIANTS[name]()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    return model.eval()


def _load(model: SMoVSR) -> SMoVSR:
    descriptor = SMoVSRArch().load(model.state_dict())
    descriptor.model.f_forward.h_clamp = model.f_forward.h_clamp
    descriptor.model.f_backward.h_clamp = model.f_backward.h_clamp
    return descriptor.model


def _bound(model: SMoVSR, state: torch.Tensor) -> torch.Tensor:
    """The clamp the model puts on a sweep's first state."""
    c = model.f_forward.h_clamp
    return state if c is None else state.clamp(-c, c)


def _per_frame(model: SMoVSR, clip: torch.Tensor) -> torch.Tensor:
    """The model's sweeps with its own modules, one frame per call."""
    count = clip.shape[1]
    feats = [model.fe(clip[:, t]) for t in range(count)]
    forward = []
    out, state = model.f_forward.step_wo_hystory(feats[0])
    state = _bound(model, state)
    forward.append(out)
    for t in range(1, count):
        out, state = model.f_forward(feats[t], state)
        forward.append(out)
    backward: list[torch.Tensor] = [torch.empty(0)] * count
    x = model.backward_cat(torch.cat([feats[-1], forward[-1]], 1))
    out, state = model.f_backward.step_wo_hystory(x)
    state = _bound(model, state)
    backward[-1] = out
    for t in range(count - 2, -1, -1):
        x = model.backward_cat(torch.cat([feats[t], forward[t]], 1))
        backward[t], state = model.f_backward(x, state)
    outputs = []
    for t in range(count):
        short = torch.nn.functional.interpolate(
            clip[:, t], scale_factor=model.scale, mode="bilinear", align_corners=False
        )
        up = model.upsampler(torch.cat([forward[t], backward[t]], 1))
        outputs.append((up + short).clamp(0, 1))
    return torch.stack(outputs, 1)


def test_load():
    assert_loads_correctly(
        SMoVSRArch(),
        *(v for k, v in VARIANTS.items() if k != "clamped"),
    )
    # the bound is no weight: loading gives upstream's default
    assert_loads_correctly(
        SMoVSRArch(), VARIANTS["clamped"], ignore_parameters={"h_clamp"}
    )


@pytest.mark.parametrize("name", list(VARIANTS))
def test_loaded_model_matches(name: str):
    model = _build(name)
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert descriptor.architecture.id == "SMoVSR"
    assert type(descriptor.model) is SMoVSR
    assert descriptor.scale == model.scale
    assert descriptor.purpose == ("Restoration" if model.scale == 1 else "SR")
    assert "temporal" in descriptor.tags
    assert descriptor.supports_half
    assert not descriptor.supports_bfloat16

    loaded = _load(model)
    clip = torch.rand(1, 3, 3, 19, 23)
    with torch.inference_mode():
        expected = model(clip)
        actual = loaded(clip)
    assert actual.shape == (1, 3, 3, 19 * model.scale, 23 * model.scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


def test_stored_merged_convolutions_are_made_again():
    model = _build("default")
    state_dict = model.state_dict()
    stale = copy.deepcopy(state_dict)
    for key, value in stale.items():
        if "eval_conv" in key:
            value.add_(0.01)
    clip = torch.rand(1, 2, 3, 16, 16)
    with torch.inference_mode():
        expected = SMoVSRArch().load(state_dict).model(clip)
        actual = SMoVSRArch().load(stale).model(clip)
    assert torch.equal(actual, expected)


def test_tags():
    assert MAIN_REGISTRY.load(_build("default").state_dict()).tags == [
        "32dim",
        "temporal",
        "rep",
    ]
    assert MAIN_REGISTRY.load(_build("plain").state_dict()).tags == [
        "32dim",
        "temporal",
    ]


@pytest.mark.parametrize("name", list(VARIANTS))
def test_other_architectures_do_not_claim_smovsr(name: str):
    state_dict = _build(name).state_dict()
    claimed = [
        support.architecture.id
        for support in MAIN_REGISTRY.architectures()
        if support.architecture.id != "SMoVSR" and support.detect(state_dict)
    ]
    assert claimed == []


@pytest.mark.parametrize("rep", [False, True])
def test_smovsr_does_not_claim_smosr(rep: bool):
    assert not SMoVSRArch().detect(SMoSR(rep=rep).state_dict())


@pytest.mark.parametrize("name", ["default", "inception", "clamped"])
def test_steps_chained_are_the_model_bit_for_bit(name: str):
    model = _build(name)
    clip = torch.rand(2, 5, 3, 21, 18)
    count = clip.shape[1]
    with torch.inference_mode():
        expected = _per_frame(model, clip)
        zero = torch.zeros(2, 32, 21, 18)
        state, feats, forward = zero, [], []
        for t in range(count):
            valid = torch.tensor([t > 0, t > 0])
            feat, out, state = encode_forward(model, clip[:, t], state, valid)
            feats.append(feat)
            forward.append(out)
        state = zero
        outputs: list[torch.Tensor] = [torch.empty(0)] * count
        for t in range(count - 1, -1, -1):
            valid = torch.tensor([t < count - 1] * 2)
            back, state = backward_step(model, feats[t], forward[t], state, valid)
            outputs[t] = head(model, clip[:, t], forward[t], back)
        actual = torch.stack(outputs, 1)
        # the model's own forward batches the frames: rounding only
        whole = model(clip).clamp(0, 1)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(actual, whole, rtol=0, atol=1e-5)


def test_window_covering_the_clip_is_the_clip():
    model = _build("default")
    clip = torch.rand(1, 6, 3, 17, 20)
    with torch.inference_mode():
        expected = _per_frame(model, clip)
        actual, _ = run_window(model, clip, 6)
        whole = run_clip(model, clip)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    torch.testing.assert_close(whole, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize(("keep", "ahead"), [(1, 2), (3, 2), (4, 5)])
def test_chained_windows_carry_the_forward_sweep_exactly(keep: int, ahead: int):
    model = _build("default")
    clip = torch.rand(1, 14, 3, 12, 15)
    count = clip.shape[1]
    with torch.inference_mode():
        expected = _per_frame(model, clip)
        state = torch.zeros(1, 32, 12, 15)
        for t in range(count):
            _, _, state = encode_forward(
                model, clip[:, t], state, torch.tensor([t > 0])
            )
        outputs = []
        past = None
        start = 0
        while True:
            end = min(start + keep + ahead, count)
            last = end == count
            taken = end - start if last else keep
            out, past = run_window(model, clip[:, start:end], taken, past)
            outputs.append(out)
            if last:
                break
            start += keep
    actual = torch.cat(outputs, 1)
    assert actual.shape == expected.shape
    # the forward sweep across windows is one sweep, bit for bit
    assert torch.equal(past[0], state)
    # the last window reads to the clip's end: its frames are the clip's
    assert torch.equal(actual[:, start:], expected[:, start:])
    # elsewhere the backward sweep starts at the window's end
    assert not torch.equal(actual[:, :start], expected[:, :start])


def test_a_carry_changes_the_window_against_a_cold_start():
    model = _build("default")
    clip = torch.rand(1, 6, 3, 10, 10)
    with torch.inference_mode():
        _, past = run_window(model, clip[:, :4], 2)
        warm, _ = run_window(model, clip[:, 2:], 4, past)
        cold, _ = run_window(model, clip[:, 2:], 4)
    assert not torch.allclose(warm, cold)


def test_window_arguments_are_checked():
    model = _build("default")
    frames = torch.rand(1, 3, 3, 8, 8)
    with torch.inference_mode():
        for keep in (0, 4):
            with pytest.raises(ValueError, match="keep must be"):
                run_window(model, frames, keep)
        _, past = run_window(model, torch.rand(1, 6, 3, 8, 8), 1)
        with pytest.raises(ValueError, match="encoded frames"):
            run_window(model, frames, 3, past)


def test_one_image_is_a_clip_of_one_frame():
    model = _build("default")
    image = torch.rand(1, 3, 13, 11)
    with torch.inference_mode():
        expected = model(image.unsqueeze(1)).squeeze(1)
        actual = model(image)
    assert torch.equal(actual, expected)


def test_half_precision():
    model = _build("default")
    clip = torch.rand(1, 3, 3, 29, 37)
    with torch.inference_mode():
        expected, _ = run_window(model, clip, 3)
        actual, _ = run_window(copy.deepcopy(model).half(), clip, 3)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.02)


def test_size_requirements():
    descriptor = SMoVSRArch().load(_build("default").state_dict())
    assert_size_requirements(descriptor)
    with torch.inference_mode():
        for height, width in ((1, 1), (5, 3), (33, 65)):
            output, _ = run_window(
                descriptor.model, torch.rand(1, 2, 3, height, width), 2
            )
            assert output.shape == (1, 2, 3, 2 * height, 2 * width)


def _load_file(path: Path) -> ImageModelDescriptor[SMoVSR]:
    descriptor = ModelLoader().load_from_file(path)
    assert isinstance(descriptor, ImageModelDescriptor)
    assert isinstance(descriptor.model, SMoVSR)
    return cast("ImageModelDescriptor[SMoVSR]", descriptor)


def _checkpoint(tmp_path: Path, name: str, network: object) -> Path:
    path = tmp_path / name
    checkpoint: dict[str, object] = {"params": _build("default").state_dict()}
    if network is not None:
        checkpoint["neosr"] = {"network_g": network, "scale": 2}
    torch.save(checkpoint, path)
    return path


def test_training_settings_set_the_bound(tmp_path: Path):
    path = _checkpoint(tmp_path, "a.pth", {"type": "VSMoSR", "h_clamp": 12.5})
    descriptor = _load_file(path)
    assert descriptor.model.f_forward.h_clamp is None
    assert apply_training_settings(descriptor, path) == 12.5
    model = descriptor.model
    assert model.f_forward.h_clamp == model.f_backward.h_clamp == 12.5
    assert model.hyperparameters["h_clamp"] == 12.5
    assert descriptor.tags == ["32dim", "temporal", "rep", "clamp12.5"]
    # applied twice: one tag
    assert apply_training_settings(descriptor, path) == 12.5
    assert descriptor.tags.count("clamp12.5") == 1


@pytest.mark.parametrize(
    "network", [None, {"type": "VSMoSR"}, {"type": "VSMoSR", "h_clamp": None}]
)
def test_without_a_recorded_bound_there_is_none(tmp_path: Path, network: object):
    path = _checkpoint(tmp_path, "b.pth", network)
    descriptor = _load_file(path)
    assert apply_training_settings(descriptor, path) is None
    assert descriptor.model.f_forward.h_clamp is None
    assert not any(tag.startswith("clamp") for tag in descriptor.tags)


def test_safetensors_has_no_training_settings(tmp_path: Path):
    path = tmp_path / "c.safetensors"
    save_file(_build("default").state_dict(), path)
    descriptor = _load_file(path)
    assert apply_training_settings(descriptor, path) is None
    assert descriptor.model.f_forward.h_clamp is None


@pytest.mark.parametrize(
    ("value", "message"),
    [("30", "must be a number"), (True, "must be a number"), (-1.0, "positive")],
)
def test_a_wrong_recorded_bound_is_an_error(
    tmp_path: Path, value: object, message: str
):
    path = _checkpoint(tmp_path, "d.pth", {"type": "VSMoSR", "h_clamp": value})
    descriptor = _load_file(path)
    with pytest.raises(ValueError, match=message):
        apply_training_settings(descriptor, path)


def _moving_clip(count: int, size: int) -> torch.Tensor:
    """A smooth pattern moving across the frame, downscaled 2x."""
    torch.manual_seed(3)
    big = torch.nn.functional.interpolate(
        torch.rand(1, 3, 48, 48), size=(256, 256), mode="bicubic", align_corners=False
    ).clamp(0, 1)
    frames = []
    for i in range(count):
        y = int(40 + 30 * math.sin(i / 9))
        x = 10 + i * 2
        frames.append(
            torch.nn.functional.avg_pool2d(
                big[:, :, y : y + 2 * size, x : x + 2 * size], 2
            )
        )
    return torch.stack(frames, 1)


@pytest.mark.skipif(
    not REAL_FILE.is_file(),
    reason="SPANDREL_TEST_SMOVSR_MODEL does not name the released file",
)
def test_real_file():
    descriptor = _load_file(REAL_FILE)
    assert descriptor.architecture.id == "SMoVSR"
    assert descriptor.scale == 2
    assert apply_training_settings(descriptor, REAL_FILE) == 30.0
    assert descriptor.tags == ["32dim", "temporal", "rep", "clamp30"]
    model = descriptor.model
    assert sum(p.numel() for p in model.parameters()) == 3_615_246

    clip = _moving_clip(40, 32)
    with torch.inference_mode():
        whole = run_clip(model, clip)
        # windows of 8 kept frames reading LOOK_AHEAD frames ahead
        outputs = []
        past = None
        start = 0
        while True:
            end = min(start + 8 + LOOK_AHEAD, 40)
            last = end == 40
            out, past = run_window(
                model, clip[:, start:end], end - start if last else 8, past
            )
            outputs.append(out)
            if last:
                break
            start += 8
    windowed = torch.cat(outputs, 1)
    assert torch.isfinite(windowed).all()
    # within the look-ahead table: 1.4/255 on the scout's clip
    assert (windowed - whole).abs().max().item() < 3 / 255
