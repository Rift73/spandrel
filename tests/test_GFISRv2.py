from __future__ import annotations

import copy

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.FDAT import FDATArch
from spandrel.architectures.FIGSR import FIGSR, FIGSRArch
from spandrel.architectures.GFISRv2 import GFISRV2, GFISRv2Arch
from spandrel.architectures.MoESR import MoESRArch
from spandrel.architectures.MoSR import MoSRArch

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)

UPSAMPLERS = (
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "transpose+conv",
    "lda",
    "pa_up",
)


def _small(**kwargs) -> GFISRV2:
    # four blocks: the Fourier branch moves through all four inception slots
    return GFISRV2(**{"dim": 24, "n_blocks": 4, "mid_dim": 16, **kwargs})


CASES = [
    pytest.param(lambda: GFISRV2(), id="defaults"),
    pytest.param(lambda: GFISRV2(scale=2), id="defaults-2x"),
    pytest.param(lambda: GFISRV2(scale=2, pixel_unshuffle=True), id="defaults-2x-pu"),
    pytest.param(lambda: _small(scale=8), id="pixelshuffledirect-8x"),
    # "conv" is a plain conv at any scale: a 1x-only upsampler
    *(
        pytest.param(
            lambda upsampler=upsampler, scale=scale: _small(
                upsampler=upsampler, scale=scale
            ),
            id=f"{upsampler}-{scale}x",
        )
        for upsampler in UPSAMPLERS
        for scale in ((1,) if upsampler == "conv" else (1, 2, 3, 4))
    ),
    # pixel_unshuffle only acts at scales 1 and 2; the upsampler then runs at 4x
    *(
        pytest.param(
            lambda upsampler=upsampler, scale=scale: _small(
                upsampler=upsampler, scale=scale, pixel_unshuffle=True
            ),
            id=f"{upsampler}-{scale}x-pu",
        )
        for upsampler in UPSAMPLERS
        if upsampler != "conv"
        for scale in (1, 2)
    ),
    pytest.param(lambda: _small(n_blocks=1), id="n_blocks-1"),
    pytest.param(lambda: _small(n_blocks=7), id="n_blocks-7"),
    pytest.param(lambda: _small(dim=32), id="dim-32"),
    pytest.param(lambda: _small(dim=64, n_blocks=2), id="dim-64"),
    pytest.param(lambda: _small(expansion_ratio=2.0), id="expansion-2"),
    pytest.param(lambda: _small(dim=32, expansion_ratio=1.5), id="expansion-1.5"),
    pytest.param(lambda: _small(in_nc=1, out_nc=1), id="gray"),
    pytest.param(lambda: _small(in_nc=4, out_nc=4), id="rgba"),
    pytest.param(lambda: _small(in_nc=3, out_nc=1), id="rgb-to-gray"),
    pytest.param(
        lambda: _small(in_nc=1, out_nc=1, scale=2, pixel_unshuffle=True),
        id="gray-2x-pu",
    ),
    pytest.param(
        lambda: _small(upsampler="pixelshuffle", mid_dim=24), id="pixelshuffle-mid-24"
    ),
    pytest.param(
        lambda: _small(upsampler="dysample", mid_dim=24), id="dysample-mid-eq-dim"
    ),
    pytest.param(lambda: _small(upsampler="lda", mid_dim=8), id="lda-mid-8"),
    pytest.param(lambda: _small(upsampler="pa_up", mid_dim=32), id="pa_up-mid-32"),
    pytest.param(lambda: _small(mid_dim=48), id="pixelshuffledirect-mid-48"),
]


def _claimants(state_dict) -> list[str]:
    return [
        support.architecture.id
        for support in MAIN_REGISTRY.architectures()
        if support.detect(state_dict)
    ]


@pytest.mark.parametrize("make", CASES)
def test_load(make):
    # hyperparameters, a safetensors round trip and the detector
    assert_loads_correctly(GFISRv2Arch(), make)


@pytest.mark.parametrize("make", CASES)
def test_loaded_model_matches(make):
    torch.manual_seed(0)
    model = make().eval()
    state_dict = model.state_dict()
    assert _claimants(state_dict) == ["GFISRv2"]

    descriptor = MAIN_REGISTRY.load(state_dict)
    loaded = descriptor.model.eval()
    hyperparameters = model.hyperparameters
    assert descriptor.architecture.id == "GFISRv2"
    assert descriptor.scale == hyperparameters["scale"]
    assert descriptor.input_channels == hyperparameters["in_nc"]
    assert descriptor.output_channels == hyperparameters["out_nc"]
    assert loaded.hyperparameters == hyperparameters
    expected_shapes = {k: v.shape for k, v in model.state_dict().items()}
    assert {k: v.shape for k, v in loaded.state_dict().items()} == expected_shapes

    # the model pads odd sizes itself; with pixel_unshuffle it needs whole
    # multiples of its padding (see test_size_requirements)
    height, width = (24, 16) if hyperparameters["pixel_unshuffle"] else (23, 18)
    image = torch.rand(1, hyperparameters["in_nc"], height, width)
    with torch.no_grad():
        expected = model(image)
        actual = loaded(image)
    scale = hyperparameters["scale"]
    out_nc = hyperparameters["out_nc"]
    assert actual.shape == (1, out_nc, height * scale, width * scale)
    torch.testing.assert_close(actual, expected)


def test_other_detectors_do_not_claim_gfisrv2():
    makes = (
        GFISRV2,
        lambda: GFISRV2(scale=2, pixel_unshuffle=True),
        lambda: _small(upsampler="dysample", scale=2),
    )
    for make in makes:
        state_dict = make().state_dict()
        for arch in (FIGSRArch(), MoSRArch(), MoESRArch(), FDATArch()):
            assert not arch.detect(state_dict), arch.id


def test_gfisrv2_does_not_claim_figsr():
    for make in (FIGSR, lambda: FIGSR(n_blocks=1, upsampler="dysample")):
        assert not GFISRv2Arch().detect(make().state_dict())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_runs_through_the_fft(dtype):
    # CPU torch.fft has neither dtype and CUDA's fp16 FFT needs power-of-two
    # sizes: FourierUnit runs its FFTs in fp32 whatever the model's dtype.
    torch.manual_seed(0)
    descriptor = MAIN_REGISTRY.load(_small(scale=2).state_dict())
    assert descriptor.supports_half
    assert descriptor.supports_bfloat16
    model = descriptor.model.eval()
    image = torch.rand(1, 3, 22, 18)
    with torch.no_grad():
        expected = model(image)
        actual = copy.deepcopy(model).to(dtype)(image.to(dtype))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0.05)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _small(scale=2), id="2x"),
        pytest.param(lambda: _small(scale=2, upsampler="lda"), id="2x-lda"),
        pytest.param(lambda: _small(scale=2, pixel_unshuffle=True), id="2x-pu"),
        pytest.param(lambda: _small(scale=1, pixel_unshuffle=True), id="1x-pu"),
    ],
)
def test_size_requirements(make):
    assert_size_requirements(GFISRv2Arch().load(make().state_dict()))
