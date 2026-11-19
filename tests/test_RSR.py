from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.RSR import (
    RSR,
    RSRArch,
    flow_working_factor,
    run_clip,
    run_window,
)
from spandrel.architectures.RSR.__arch.rsr_arch import ContextGate, rsr_compact
from spandrel.architectures.WSR.__arch.wsr_arch import wsr_compact

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)

# The registered preset at every scale, then non-default shapes that change the
# weights.
VARIANTS = {
    **{
        f"Compact-x{scale}": (lambda scale=scale: rsr_compact(scale=scale))
        for scale in (1, 2, 4)
    },
    "RSR-D5-S2-H2": lambda: RSR(
        width=16, hidden=32, context_width=16, depth=5, gates=2, heads=2, scale=2
    ),
}

# The temporal read: the neighbours' projections and the null token.
TEMPORAL = (
    "attention.key_value_past.",
    "attention.key_value_future.",
    "attention.null_",
)


def _build(name: str, temporal: bool = True) -> RSR:
    """Give every zero-initialized weight some value so that the whole model is
    exercised; with temporal False, the temporal read keeps its initial zero
    values."""
    torch.manual_seed(0)
    model = VARIANTS[name]()
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if temporal or not key.startswith(TEMPORAL):
                parameter.add_(torch.randn_like(parameter) * 0.05 * (parameter == 0))
    return model.eval()


def _deployed(name: str) -> RSR:
    return RSRArch().load(_build(name).state_dict()).model


def test_load():
    assert_loads_correctly(RSRArch(), *VARIANTS.values())


@pytest.mark.parametrize("folded", [False, True], ids=["training", "folded"])
@pytest.mark.parametrize("name", list(VARIANTS))
def test_loaded_model_matches(name: str, folded: bool):
    model = _build(name)
    deployed = copy.deepcopy(model).prepare_for_export()
    source = deployed if folded else model
    descriptor = MAIN_REGISTRY.load(source.state_dict())

    assert descriptor.architecture.id == "RSR"
    assert type(descriptor.model) is RSR
    assert descriptor.model.hyperparameters == model.hyperparameters
    assert descriptor.scale == model.scale
    assert descriptor.purpose == ("Restoration" if model.scale == 1 else "SR")
    assert "temporal" in descriptor.tags
    assert descriptor.supports_half
    assert descriptor.supports_bfloat16

    actual_state = descriptor.model.state_dict()
    expected_state = deployed.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        assert torch.equal(actual_state[key], value), key

    frames = torch.rand(1, 3, 3, 29, 37)
    with torch.no_grad():
        expected = model(frames)
        actual = descriptor.model(frames)
    assert actual.shape == (1, 3, 3, 29 * model.scale, 37 * model.scale)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_compact_tag():
    assert MAIN_REGISTRY.load(_build("Compact-x2").state_dict()).tags == [
        "Compact",
        "32nf",
        "9nb",
        "temporal",
    ]


@pytest.mark.parametrize("folded", [False, True], ids=["training", "folded"])
def test_registry_tells_rsr_from_wsr(folded: bool):
    rsr = _build("Compact-x2")
    wsr = wsr_compact(scale=2).eval()
    if folded:
        rsr.prepare_for_export()
        wsr.prepare_for_export()
    assert MAIN_REGISTRY.load(rsr.state_dict()).architecture.id == "RSR"
    assert MAIN_REGISTRY.load(wsr.state_dict()).architecture.id == "WSR"
    assert not RSRArch().detect(wsr.state_dict())


@pytest.mark.parametrize("folded", [False, True], ids=["training", "folded"])
@pytest.mark.parametrize("name", list(VARIANTS))
def test_other_architectures_do_not_claim_rsr(name: str, folded: bool):
    model = _build(name)
    if folded:
        model.prepare_for_export()
    state_dict = model.state_dict()
    claimed = [
        support.architecture.id
        for support in MAIN_REGISTRY.architectures()
        # WSR comes after RSR: an RSR file holds every WSR weight.
        if support.architecture.id != "RSR"
        and support.architecture.id != "WSR"
        and support.detect(state_dict)
    ]
    assert claimed == []


def test_zero_temporal_read_is_the_per_frame_model():
    model = RSRArch().load(_build("Compact-x2", temporal=False).state_dict()).model
    frames = torch.rand(1, 4, 3, 29, 37)
    with torch.inference_mode():
        clip = run_clip(model, frames)
        alone = torch.stack([model(frames[:, t]) for t in range(4)], 1)
    torch.testing.assert_close(clip, alone, rtol=0, atol=1e-5)


def test_one_image_is_one_frame_without_neighbours():
    descriptor = RSRArch().load(_build("Compact-x2").state_dict())
    image = torch.rand(1, 3, 29, 37)
    with torch.inference_mode():
        out = descriptor(image)
        window, _ = run_window(descriptor.model, image[:, None], 1)
    assert out.shape == (1, 3, 58, 74)
    torch.testing.assert_close(out, window[:, 0].clamp(0, 1), rtol=0, atol=0)


def test_window_covering_the_clip_is_the_clip():
    model = _deployed("Compact-x2")
    frames = torch.rand(2, 5, 3, 29, 37)
    with torch.inference_mode():
        clip = run_clip(model, frames)
        window, (state, encoded, previous) = run_window(model, frames, 5)
    assert window.dtype == torch.float32
    assert window.shape == (2, 5, 3, 58, 74)
    assert state.shape == (2, 32, 29, 37)
    assert encoded == ()
    assert previous is None
    torch.testing.assert_close(window, clip, rtol=0, atol=0)


@pytest.mark.parametrize(("size", "keep"), [(4, 3), (3, 1)])
def test_chained_windows_are_infer(size: int, keep: int):
    # The app's pattern: windows of `size` frames keep all but the last
    # size - keep, the next window starts after the kept frames, and the last
    # window keeps everything up to the clip's end.
    model = _deployed("Compact-x2")
    frames = torch.rand(1, 8, 3, 29, 37)
    outputs = []
    past = None
    with torch.inference_mode():
        expected = model.infer(frames, chunk=keep, lookahead=size - keep)
        for start in range(0, 8, keep):
            window = frames[:, start : start + size]
            count = min(keep, window.shape[1])
            output, past = run_window(model, window, count, past)
            outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=0, atol=0)


def test_a_carried_state_changes_the_window_against_a_cold_start():
    model = _deployed("Compact-x2")
    frames = torch.rand(1, 6, 3, 29, 37)
    with torch.inference_mode():
        _, past = run_window(model, frames[:, :4], 3)
        carried, _ = run_window(model, frames[:, 3:], 3, past)
        cold, _ = run_window(model, frames[:, 3:], 3)
    assert (carried - cold).abs().amax() > 1e-3


def test_window_arguments_are_checked():
    model = _deployed("Compact-x2")
    frames = torch.rand(1, 3, 3, 16, 16)
    with torch.inference_mode():
        with pytest.raises(ValueError, match="keep"):
            run_window(model, frames, 4)
        with pytest.raises(ValueError, match="keep"):
            run_window(model, frames, 0)
        _, past = run_window(model, torch.rand(1, 5, 3, 16, 16), 1)
        with pytest.raises(ValueError, match="encoded frames"):
            run_window(model, frames, 3, past)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_precision(dtype: torch.dtype):
    model = _deployed("Compact-x2")
    frames = torch.rand(1, 3, 3, 29, 37)
    with torch.inference_mode():
        expected, _ = run_window(model, frames, 3)
        actual, _ = run_window(model.to(dtype), frames, 3)
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    tolerance = 0.02 if dtype == torch.float16 else 0.1
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)


def test_size_requirements():
    descriptor = RSRArch().load(_build("Compact-x2").state_dict())
    assert_size_requirements(descriptor)
    with torch.inference_mode():
        for height, width in ((1, 1), (5, 3), (33, 65)):
            output, _ = run_window(
                descriptor.model, torch.rand(1, 2, 3, height, width), 2
            )
            assert output.shape == (1, 2, 3, 2 * height, 2 * width)


# The aligned read: a flow-aligned read of the neighbours' states, blended in.


def _aligned(blend: bool = True) -> RSR:
    """An aligned Compact x2 with random weights everywhere; with blend False the
    blends keep their initial zero values and the read adds nothing."""
    torch.manual_seed(0)
    model = rsr_compact(scale=2, temporal_read="aligned")
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if blend or not key.startswith("blend_"):
                parameter.add_(torch.randn_like(parameter) * 0.05 * (parameter == 0))
        model.flow_loaded.fill_(True)
    return model.eval()


def test_aligned_is_detected_as_rsr():
    state_dict = _aligned().state_dict()
    assert "attention.null_key" not in state_dict
    descriptor = MAIN_REGISTRY.load(state_dict)
    assert descriptor.architecture.id == "RSR"
    assert descriptor.model.temporal_read == "aligned"
    assert "aligned" in descriptor.tags
    assert descriptor.scale == 2
    assert descriptor.model.blend_forward.weight.shape == (32, 96, 3, 3)
    block = RSRArch().load(_build("Compact-x2").state_dict())
    assert block.model.temporal_read == "block"
    assert "aligned" not in block.tags


def test_aligned_folded_equals_unfolded():
    model = _aligned()
    folded = RSRArch().load(model.state_dict()).model
    assert folded.deploy
    frames = torch.rand(1, 3, 3, 32, 40)
    with torch.inference_mode():
        torch.testing.assert_close(
            run_clip(folded, frames), run_clip(model, frames), rtol=0, atol=1e-4
        )


def test_aligned_zero_blends_are_the_per_frame_model():
    model = RSRArch().load(_aligned(blend=False).state_dict()).model
    frames = torch.rand(1, 3, 3, 32, 40)
    with torch.inference_mode():
        clip = run_clip(model, frames)
        alone = torch.stack([model(frames[:, t]) for t in range(3)], 1)
    torch.testing.assert_close(clip, alone, rtol=0, atol=1e-5)


def test_aligned_window_covering_the_clip_is_the_clip():
    model = RSRArch().load(_aligned().state_dict()).model
    frames = torch.rand(1, 4, 3, 32, 40)
    with torch.inference_mode():
        clip = run_clip(model, frames)
        alone = torch.stack([model(frames[:, t]) for t in range(4)], 1)
        window, (_, encoded, previous) = run_window(model, frames, 4)
    assert (clip - alone).abs().amax() > 1e-3
    assert encoded == ()
    assert previous is not None
    torch.testing.assert_close(previous, frames[:, 3:4], rtol=0, atol=0)
    torch.testing.assert_close(window, clip, rtol=0, atol=0)


@pytest.mark.parametrize(("size", "keep"), [(4, 3), (3, 1)])
def test_aligned_chained_windows_are_infer(size: int, keep: int):
    model = RSRArch().load(_aligned().state_dict()).model
    frames = torch.rand(1, 6, 3, 32, 40)
    outputs = []
    past = None
    with torch.inference_mode():
        expected = model.infer(frames, chunk=keep, lookahead=size - keep)
        for start in range(0, 6, keep):
            window = frames[:, start : start + size]
            count = min(keep, window.shape[1])
            output, past = run_window(model, window, count, past)
            outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=0, atol=0)


def test_aligned_reduced_precision():
    model = RSRArch().load(_aligned().state_dict()).model
    frames = torch.rand(1, 3, 3, 32, 40)
    with torch.inference_mode():
        expected, _ = run_window(model, frames, 3)
        actual, _ = run_window(model.to(torch.bfloat16), frames, 3)
    assert actual.dtype == torch.float32
    assert model.flow_loaded.dtype == torch.bool
    assert all(p.dtype == torch.float32 for p in model.flow.parameters())
    assert model.blend_forward.weight.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.1)


def test_flow_working_factor():
    assert flow_working_factor(240, 320) == 1
    assert flow_working_factor(1080, 1920) == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eval_gate_upsamples_as_a_scale_factor_of_4(dtype: torch.dtype):
    # The eval gate's interpolation by sizes gives the numbers of scale_factor=4.0
    # bit for bit, on a frame not divisible by 4 (the map is cropped to it).
    torch.manual_seed(0)
    gate = ContextGate(8, 16).eval()
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.normal_(0, 0.5)
    gate = gate.to(dtype)
    x = torch.rand(1, 8, 29, 37, dtype=dtype)
    context = torch.rand(1, 16, 8, 10, dtype=dtype)
    with torch.no_grad():
        maps = torch.nn.functional.interpolate(
            gate.project(context),
            scale_factor=4.0,
            mode="bilinear",
            align_corners=False,
        )
        gamma, beta = maps[..., :29, :37].chunk(2, dim=1)
        expected = x * (2.0 * torch.sigmoid(gamma + gate.local_scale * x)) + beta
        actual = gate(x, context)
    assert torch.equal(actual, expected)


def test_onnx_export_pads_without_modulo(tmp_path: Path):
    # The pads to a multiple of 4 as WSR's: no Neg or Mod, which TensorRT cannot bound
    # for dynamic shapes. A 4-D input is one frame without neighbours.
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "rsr.onnx"
    torch.onnx.export(
        _deployed("Compact-x2"),
        (torch.rand(1, 3, 64, 64),),
        str(path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
        dynamo=False,
    )
    graph = onnx.load(str(path)).graph
    assert "Pad" in {node.op_type for node in graph.node}
    assert not {"Neg", "Mod"} & {node.op_type for node in graph.node}


def _channels_last_clip(clip: torch.Tensor) -> torch.Tensor:
    """The same B,T,3,H,W values with each frame channels-last, as a window's stack on a
    GPU holds them."""
    batch, count, channels, height, width = clip.shape
    stacked = clip.flatten(0, 1).contiguous(memory_format=torch.channels_last)
    return stacked.reshape(batch, count, channels, height, width)


def test_aligned_flow_takes_nchw_frames_whatever_their_layout():
    # From channels-last frames cuDNN picks, for the flow's 130-channel convolution,
    # an algorithm of 4.3-4.7 GiB of workspace: the flow is handed NCHW frames.
    model = RSRArch().load(_aligned().state_dict()).model
    seen: list[bool] = []

    def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        seen.extend(arg.is_contiguous() for arg in args)

    handle = model.flow.register_forward_pre_hook(hook)
    torch.manual_seed(1)
    clip = _channels_last_clip(torch.rand(1, 3, 3, 64, 96))
    assert clip[0].is_contiguous(memory_format=torch.channels_last)
    assert not clip[0, 0].is_contiguous()
    try:
        with torch.inference_mode():
            model.guidance(clip)
    finally:
        handle.remove()
    assert seen and all(seen)


def test_aligned_guidance_is_the_same_for_channels_last_frames():
    model = RSRArch().load(_aligned().state_dict()).model
    torch.manual_seed(1)
    clip = torch.rand(1, 3, 3, 64, 96)
    with torch.inference_mode():
        expected = model.guidance(clip)
        actual = model.guidance(_channels_last_clip(clip))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "stripped",
    ["no flag", "no flow weights", "no flow weights and no flag", "flag false"],
)
def test_aligned_without_its_flow_is_refused_as_rsr(stripped: str):
    # The flow's weights are most of an aligned file: stripped, it is still told apart
    # from WSR, and refused with a message that names what is missing.
    state = _aligned().state_dict()
    if stripped == "flag false":
        state["flow_loaded"] = torch.tensor(False)
    if "no flag" in stripped:
        del state["flow_loaded"]
    if "no flow weights" in stripped:
        state = {k: v for k, v in state.items() if not k.startswith("flow.")}
    with pytest.raises(
        ValueError, match="RSR model with optical flow holds no optical"
    ):
        MAIN_REGISTRY.load(state)


def test_a_complete_aligned_file_loads():
    descriptor = MAIN_REGISTRY.load(_aligned().state_dict())
    assert descriptor.architecture.id == "RSR" and "aligned" in descriptor.tags


def test_aligned_carry_holds_its_frame_not_the_window():
    model = RSRArch().load(_aligned().state_dict()).model
    frames = torch.rand(1, 3, 3, 32, 32)
    with torch.inference_mode():
        _, (_, _, last) = run_window(model, frames, 2)
    assert last is not None
    assert torch.equal(last, frames[:, 1:2])
    assert last.untyped_storage().nbytes() == last.numel() * last.element_size()
