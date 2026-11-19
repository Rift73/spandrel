from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.MoSR import MoSR
from spandrel.architectures.RSR import RSR
from spandrel.architectures.SPAN import SPAN
from spandrel.architectures.WSR import WESR, WideContextSR, WSRArch
from spandrel.architectures.WSR.__arch.wsr_arch import (
    ContextGate,
    wsr,
    wsr_compact,
    wsr_mobile,
    wsr_superultracompact,
    wsr_ultracompact,
)

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)

PRESETS = {
    "Default": wsr,
    "Compact": wsr_compact,
    "UltraCompact": wsr_ultracompact,
    "SuperUltraCompact": wsr_superultracompact,
    "Mobile": wsr_mobile,
}

# Every registered preset at every scale (the source's training configs are
# these presets at 4x), then non-default values of every option that changes
# the weights.
VARIANTS = {
    **{
        f"{name}-x{scale}": (lambda factory=factory, scale=scale: factory(scale=scale))
        for name, factory in PRESETS.items()
        for scale in (1, 2, 4)
    },
    "WESR-defaults": lambda: WESR(),
    "WESR-no-attention": lambda: WESR(use_xsa=False, scale=2),
    "WESR-D4-S2-H2": lambda: WESR(
        width=16, hidden=32, context_width=16, depth=4, gates=2, heads=2, scale=2
    ),
    "WESR-D5-S5-H3": lambda: WESR(
        width=24, hidden=48, context_width=24, depth=5, gates=5, heads=3, scale=1
    ),
    "WESR-D3-S1-no-attention": lambda: WESR(
        width=8,
        hidden=8,
        context_width=8,
        depth=3,
        gates=1,
        heads=1,
        use_xsa=False,
        scale=4,
    ),
    "WideContextSR-defaults": lambda: WideContextSR(),
    "WideContextSR-D3-S3": lambda: WideContextSR(
        width=16, hidden=16, depth=3, sft_count=3, context_width=8, scale=2
    ),
    "WideContextSR-D6-S1": lambda: WideContextSR(
        width=8, hidden=24, depth=6, sft_count=1, context_width=16, scale=1
    ),
}


def _activate(model: WESR | WideContextSR) -> WESR | WideContextSR:
    """Give every zero-initialized weight (context merge and gates, attention
    values) some value so that the whole model is exercised."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05 * (parameter == 0))
    return model


def _build(name: str) -> WESR | WideContextSR:
    torch.manual_seed(0)
    return _activate(VARIANTS[name]()).eval()


def test_load():
    assert_loads_correctly(WSRArch(), *VARIANTS.values())


@pytest.mark.parametrize("folded", [False, True], ids=["training", "folded"])
@pytest.mark.parametrize("name", list(VARIANTS))
def test_loaded_model_matches(name: str, folded: bool):
    model = _build(name)
    deployed = copy.deepcopy(model).prepare_for_export()
    source = deployed if folded else model
    descriptor = MAIN_REGISTRY.load(source.state_dict())

    assert descriptor.architecture.id == "WSR"
    assert type(descriptor.model) is type(model)
    assert descriptor.model.hyperparameters == model.hyperparameters
    assert descriptor.scale == model.scale
    assert descriptor.input_channels == 3
    assert descriptor.output_channels == 3
    assert descriptor.purpose == ("Restoration" if model.scale == 1 else "SR")

    # The loaded model runs folded: compare it with the folded source.
    actual_state = descriptor.model.state_dict()
    expected_state = deployed.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        assert actual_state[key].shape == value.shape, key
        assert torch.equal(actual_state[key], value), key

    image = torch.rand(1, 3, 29, 37)
    with torch.no_grad():
        expected = model(image)
        actual = descriptor.model(image)
    assert actual.shape == (1, 3, 29 * model.scale, 37 * model.scale)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", ["Default-x4", "Compact-x2", "Mobile-x2"])
def test_preset_tags(name: str):
    descriptor = MAIN_REGISTRY.load(_build(name).state_dict())
    assert descriptor.tags[0] == name.split("-")[0]


@pytest.mark.parametrize("name", ["Default-x2", "Compact-x1", "Compact-x4"])
def test_attention_is_halo(name: str):
    # RSR reads one frame without neighbours with the same Halo attention: 32x32
    # query tiles reading a 64x64 donor region across the tile borders.
    model = _build(name)
    options = {k: v for k, v in model.hyperparameters.items() if k != "use_xsa"}
    rsr = RSR(**options).eval()
    missing, unexpected = rsr.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected
    assert all(key.startswith("attention.") for key in missing)
    descriptor = MAIN_REGISTRY.load(model.state_dict())

    image = torch.rand(1, 3, 45, 70)
    with torch.no_grad():
        expected = rsr(image)
        actual = descriptor.model(image)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("folded", [False, True], ids=["training", "folded"])
@pytest.mark.parametrize("name", list(VARIANTS))
def test_other_architectures_do_not_claim_wsr(name: str, folded: bool):
    model = _build(name)
    if folded:
        model.prepare_for_export()
    state_dict = model.state_dict()
    claimed = [
        support.architecture.id
        for support in MAIN_REGISTRY.architectures()
        if support.architecture.id != "WSR" and support.detect(state_dict)
    ]
    assert claimed == []


def test_wsr_does_not_claim_other_architectures():
    arch = WSRArch()
    assert not arch.detect(SPAN(num_in_ch=3, num_out_ch=3).state_dict())
    assert not arch.detect(SPAN(num_in_ch=3, num_out_ch=3, upscale=1).state_dict())
    for upsampler in ("ps", "dys", "gps"):
        model = MoSR(n_block=2, dim=32, upsampler=upsampler)
        assert not arch.detect(model.state_dict())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", ["Default-x2", "Compact-x4", "UltraCompact-x2"])
def test_reduced_precision(name: str, dtype: torch.dtype):
    model = _build(name)
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert descriptor.supports_half
    assert descriptor.supports_bfloat16

    image = torch.rand(1, 3, 29, 37)
    with torch.no_grad():
        expected = model(image)
        actual = descriptor.model.to(dtype)(image.to(dtype)).float()
    assert torch.isfinite(actual).all()
    tolerance = 0.02 if dtype == torch.float16 else 0.1
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)


@pytest.mark.parametrize("name", ["Compact-x2", "Mobile-x4"])
def test_size_requirements(name: str):
    assert_size_requirements(MAIN_REGISTRY.load(_build(name).state_dict()))


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


def test_onnx_export_resizes_by_sizes_and_pads_without_modulo(tmp_path: Path):
    # A Resize with sizes, not scales, has an output shape TensorRT's shape inference
    # bounds for dynamic inputs (Build Engine's Dynamic shapes).
    onnx = pytest.importorskip("onnx")
    descriptor = MAIN_REGISTRY.load(_build("Compact-x2").state_dict())
    path = tmp_path / "wsr.onnx"
    torch.onnx.export(
        descriptor.model,
        (torch.rand(1, 3, 64, 64),),
        str(path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
        dynamo=False,
    )
    graph = onnx.load(str(path)).graph
    resizes = [node for node in graph.node if node.op_type == "Resize"]
    assert len(resizes) == 3
    for node in resizes:
        # inputs: X, roi, scales, sizes
        assert len(node.input) == 4 and node.input[2] == "" and node.input[3] != ""
    # The pads to a multiple of 4 as (n + 3) // 4 * 4 - n: TensorRT cannot bound the
    # context map's size through the Neg and Mod of (-n) % 4.
    assert not {"Neg", "Mod"} & {node.op_type for node in graph.node}


def test_pads_to_4_are_the_integers_of_the_negated_modulo():
    # Eager runs on Python integers: the same pads give the same output bit for bit.
    for n in range(1, 257):
        assert (n + 3) // 4 * 4 - n == (-n) % 4
