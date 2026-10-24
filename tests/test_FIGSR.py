from __future__ import annotations

import copy
from typing import Any

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


def _small(**kwargs: Any) -> FIGSR:
    options: dict[str, Any] = {"dim": 24, "n_blocks": 2, "gc": 4, "mid_dim": 16}
    return FIGSR(**(options | kwargs))


# The basicsr, neosr and traiNNer-redux configs of the source repository all
# train the default FIGSR at scale 4.
CASES = [
    pytest.param(lambda: FIGSR(), id="defaults-and-configs"),
    pytest.param(lambda: FIGSR(scale=2), id="defaults-2x"),
    # 8x: pixelshuffle, nearest+conv and pa_up repeat their 2x stage;
    # transpose+conv stops at 4x
    *(
        pytest.param(
            lambda upsampler=upsampler: _small(upsampler=upsampler, scale=8),
            id=f"{upsampler}-8x",
        )
        for upsampler in UPSAMPLERS
        if upsampler not in ("conv", "transpose+conv")
    ),
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
    pytest.param(lambda: _small(n_blocks=1), id="n_blocks-1"),
    pytest.param(lambda: _small(n_blocks=3), id="n_blocks-3"),
    pytest.param(lambda: _small(dim=32), id="dim-32"),
    pytest.param(lambda: _small(expansion_ratio=2.0), id="expansion-2"),
    pytest.param(lambda: _small(dim=32, expansion_ratio=1.5), id="expansion-1.5"),
    pytest.param(lambda: _small(gc=2), id="gc-2"),
    pytest.param(lambda: _small(square_kernel_size=7), id="square-kernel-7"),
    pytest.param(lambda: _small(band_kernel_size=11), id="band-kernel-11"),
    pytest.param(
        lambda: _small(upsampler="pixelshuffle", mid_dim=24), id="pixelshuffle-mid-24"
    ),
    pytest.param(
        lambda: _small(upsampler="dysample", mid_dim=24), id="dysample-mid-24"
    ),
    pytest.param(
        lambda: _small(upsampler="dysample", mid_dim=24, dim=24),
        id="dysample-mid-eq-dim",
    ),
    pytest.param(lambda: _small(upsampler="lda", mid_dim=8), id="lda-mid-8"),
    pytest.param(lambda: _small(upsampler="pa_up", mid_dim=24), id="pa_up-mid-24"),
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
    assert_loads_correctly(FIGSRArch(), make)


@pytest.mark.parametrize("make", CASES)
def test_loaded_model_matches(make):
    torch.manual_seed(0)
    model = make().eval()
    state_dict = model.state_dict()
    assert _claimants(state_dict) == ["FIGSR"]

    descriptor = MAIN_REGISTRY.load(state_dict)
    loaded = descriptor.model.eval()
    hyperparameters = model.hyperparameters
    assert descriptor.architecture.id == "FIGSR"
    assert descriptor.scale == hyperparameters["scale"]
    assert descriptor.input_channels == hyperparameters["in_nc"]
    assert descriptor.output_channels == hyperparameters["out_nc"]
    assert loaded.hyperparameters == hyperparameters
    expected_shapes = {k: v.shape for k, v in model.state_dict().items()}
    assert {k: v.shape for k, v in loaded.state_dict().items()} == expected_shapes

    # odd sizes: the model pads to even sizes and crops the output
    image = torch.rand(1, 3, 23, 18)
    with torch.no_grad():
        expected = model(image)
        actual = loaded(image)
    scale = hyperparameters["scale"]
    assert actual.shape == (1, 3, 23 * scale, 18 * scale)
    torch.testing.assert_close(actual, expected)


def test_channels_load_from_the_weights():
    # shift and scale_norm hold three channels, so only RGB models run; the
    # loader still reads in_nc and out_nc from the weights
    assert_loads_correctly(
        FIGSRArch(),
        lambda: _small(in_nc=1, out_nc=1),
        lambda: _small(in_nc=4, out_nc=3),
    )
    descriptor = MAIN_REGISTRY.load(_small(in_nc=1, out_nc=1).state_dict())
    assert descriptor.input_channels == 1
    assert descriptor.output_channels == 1


def test_other_detectors_do_not_claim_figsr():
    for make in (FIGSR, lambda: _small(upsampler="dysample", scale=2)):
        state_dict = make().state_dict()
        for arch in (GFISRv2Arch(), MoSRArch(), MoESRArch(), FDATArch()):
            assert not arch.detect(state_dict), arch.id


def test_figsr_does_not_claim_gfisrv2():
    for make in (GFISRV2, lambda: GFISRV2(scale=2, pixel_unshuffle=True)):
        assert not FIGSRArch().detect(make().state_dict())


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


def test_size_requirements():
    assert_size_requirements(FIGSRArch().load(_small(scale=2).state_dict()))
    assert_size_requirements(
        FIGSRArch().load(_small(scale=2, upsampler="lda").state_dict())
    )
