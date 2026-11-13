import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from spandrel import MAIN_REGISTRY, ImageModelDescriptor
from spandrel.architectures.DUAL.__arch.dual_arch import dual_light
from spandrel.architectures.RDUAL import RDUAL, RDUALArch, run_clip, run_window

from .util import assert_loads_correctly, skip_if_unchanged

skip_if_unchanged(__file__)


def _model(**kwargs) -> RDUAL:
    model = RDUAL(use_checkpoint=False, **kwargs)
    # Stands in for real flow weights, which a saved RDUAL generator always holds.
    model.get_buffer("flow_loaded").fill_(True)
    return model


def test_load():
    assert_loads_correctly(
        RDUALArch(),
        lambda: _model(scale=2),
        lambda: _model(scale=4),
        lambda: _model(scale=1),
        lambda: _model(scale=2, dynamic_experts=2),
    )


def test_registry_tells_rdual_from_dual():
    assert MAIN_REGISTRY.load(_model(scale=2).state_dict()).architecture.id == "RDUAL"
    assert MAIN_REGISTRY.load(dual_light(2).state_dict()).architecture.id == "DUAL"


def test_weights_without_flow_are_rejected():
    model = _model(scale=2)
    model.get_buffer("flow_loaded").fill_(False)
    with pytest.raises(ValueError, match="optical flow"):
        RDUALArch().load(model.state_dict())


def test_zero_temporal_stage_is_dual_light_on_every_frame():
    # A DUAL Light checkpoint in RDUAL: the temporal stage starts at zero output, so
    # every frame of a clip must come out as DUAL Light alone makes it.
    torch.manual_seed(0)
    dual = dual_light(2).eval()
    for parameter in dual.parameters():
        parameter.data.normal_(0, 0.02)
    rdual = _model(scale=2)
    missing, unexpected = rdual.load_state_dict(dual.state_dict(), strict=False)
    assert not unexpected
    assert all(key.startswith(("temporal.", "flow.", "flow_loaded")) for key in missing)
    rdual_descriptor = RDUALArch().load(rdual.state_dict())
    dual_descriptor = MAIN_REGISTRY.load(dual.state_dict())
    assert isinstance(dual_descriptor, ImageModelDescriptor)

    frames = torch.rand(1, 2, 3, 32, 40)
    with torch.inference_mode():
        clip = run_clip(rdual_descriptor.model, frames)
        alone = torch.stack([dual_descriptor(frames[:, t]) for t in range(2)], 1)
    assert clip.shape == (1, 2, 3, 64, 80)
    # The descriptor's call clamps to 0..1; run_clip returns the model's own values.
    torch.testing.assert_close(clip.clamp(0, 1), alone, atol=2e-5, rtol=0)


def test_one_image_is_a_clip_of_one_frame():
    descriptor = RDUALArch().load(_model(scale=2).state_dict())
    image = torch.rand(1, 3, 32, 40)
    with torch.inference_mode():
        out = descriptor(image)
        clip = run_clip(descriptor.model, image[:, None])[:, 0]
    assert out.shape == (1, 3, 64, 80)
    torch.testing.assert_close(out, clip.clamp(0, 1), atol=0, rtol=0)


def _temporal_model() -> RDUAL:
    # The temporal stage starts at zero output; random weights make every frame
    # read its neighbors and the carried state.
    torch.manual_seed(0)
    model = _model(scale=2).eval()
    with torch.no_grad():
        for parameter in model.temporal.parameters():
            parameter.normal_(0, 0.05)
    return model


def test_chained_windows_reaching_the_end_are_the_whole_clip():
    model = _temporal_model()
    frames = torch.rand(1, 7, 3, 32, 40)
    with torch.inference_mode():
        whole = model(frames)
        first, state = run_window(model, frames, 3)
        second, _ = run_window(model, frames[:, 3:], 4, (state, frames[:, 2]))
    assert first.shape == (1, 3, 3, 64, 80)
    assert second.shape == (1, 4, 3, 64, 80)
    torch.testing.assert_close(torch.cat((first, second), 1), whole)


def test_a_carried_state_changes_the_window_against_a_cold_start():
    model = _temporal_model()
    frames = torch.rand(1, 6, 3, 32, 40)
    with torch.inference_mode():
        _, state = run_window(model, frames[:, :3], 3)
        carried, _ = run_window(model, frames[:, 3:], 2, (state, frames[:, 2]))
        cold, _ = run_window(model, frames[:, 3:], 2)
    assert carried.shape == cold.shape == (1, 2, 3, 64, 80)
    assert (carried - cold).abs().amax() > 1e-3


@pytest.mark.parametrize("keep", [0, 4])
def test_a_window_keeps_one_to_all_of_its_frames(keep):
    with pytest.raises(ValueError, match="keep must be"):
        _temporal_model().infer_window(torch.rand(1, 3, 3, 32, 40), keep)


class _TritonError(Exception):
    pass


_TritonError.__module__ = "triton.compiler.errors"


def test_a_triton_failure_switches_to_the_pytorch_temporal_path(monkeypatch):
    model = _model(scale=2)
    calls: list[str] = []

    def forward(model: RDUAL, frames: torch.Tensor) -> torch.Tensor:
        calls.append(model.temporal.backend)
        if model.temporal.backend == "auto":
            raise RuntimeError("kernel launch") from _TritonError("ast.Num")
        return frames

    monkeypatch.setattr("spandrel.architectures.RDUAL._forward", forward)
    frames = torch.rand(1, 2, 3, 16, 16)
    assert run_clip(model, frames) is frames
    assert calls == ["auto", "reference"]
    assert model.temporal.backend == "reference"


def test_other_failures_are_raised(monkeypatch):
    model = _model(scale=2)

    def forward(*_: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr("spandrel.architectures.RDUAL._forward", forward)
    with pytest.raises(torch.OutOfMemoryError):
        run_clip(model, torch.rand(1, 2, 3, 16, 16))
    assert model.temporal.backend == "auto"


def test_importing_spandrel_leaves_the_architecture_module_unloaded():
    # It imports Triton and Flex attention: only an RDUAL model pays for them.
    module = "spandrel.architectures.RDUAL.__arch.rdual_arch"
    code = "\n".join(
        [
            "import sys, spandrel",
            "from spandrel.architectures import RDUAL",
            f"assert {module!r} not in sys.modules",
            "assert RDUAL.RDUAL.__name__ == 'RDUAL'",
            f"assert {module!r} in sys.modules",
        ]
    )
    spandrel = Path(__file__).parent.parent / "libs" / "spandrel"
    env = {**os.environ, "PYTHONPATH": str(spandrel)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=240)
