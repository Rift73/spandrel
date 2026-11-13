from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.__arch_helpers import flash_window
from spandrel.architectures.ATD import ATD
from spandrel.architectures.DRCT import DRCT
from spandrel.architectures.DRFT import DRFT
from spandrel.architectures.HAT import HAT
from spandrel.architectures.HEART import HEART, HEARTArch
from spandrel.architectures.HEART.__arch import heart_arch
from spandrel.architectures.SST.__arch import sst_arch
from spandrel.architectures.Swin2SR import Swin2SR
from spandrel.architectures.SwinIR import SwinIR

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)


def _small(**kwargs: Any) -> HEART:
    options: dict[str, Any] = {
        "embed_dim": 24,
        "depths": (2, 2),
        "num_heads": (2, 2),
        "num_feat": 16,
        "upscale": 2,
        "squeeze_factor": 6,
    }
    return HEART(**(options | kwargs))


# The released models (huggingface.co/Phips/HEART): heart_4x_pretrain,
# heart_4x_otf_v2, heart_4x_otf_gan and 4x_HEART_pretrain_cc0 use the defaults
# at 4x, heart_2x the defaults at 2x.
CASES = [
    pytest.param(lambda: HEART(upscale=4), id="released-4x"),
    pytest.param(lambda: HEART(upscale=2), id="released-2x"),
    pytest.param(lambda: _small(upscale=1), id="1x"),
    pytest.param(lambda: _small(upscale=3), id="3x"),
    pytest.param(lambda: _small(upscale=8), id="8x"),
    pytest.param(lambda: _small(attention_freq=1), id="attention-every-block"),
    pytest.param(
        lambda: _small(depths=(3, 3), attention_freq=3), id="attention-every-third"
    ),
    pytest.param(
        lambda: _small(depths=(5,), num_heads=(4,), attention_freq=5),
        id="one-attention-block",
    ),
    pytest.param(lambda: _small(depths=(3, 1), num_heads=(4, 6)), id="uneven"),
    pytest.param(lambda: _small(rank=4), id="rank-4"),
    pytest.param(lambda: _small(rib_n_freqs=6, rib_hidden_dim=16), id="rib-small"),
    pytest.param(lambda: _small(rib_n_freqs=0), id="rib-no-freqs"),
    pytest.param(lambda: _small(mlp_ratio=4.0), id="mlp-4"),
    pytest.param(lambda: _small(compress_ratio=4, squeeze_factor=12), id="cab"),
    pytest.param(lambda: _small(resi_connection="identity"), id="identity"),
    pytest.param(lambda: _small(patch_norm=False), id="no-patch-norm"),
    pytest.param(lambda: _small(ape=True, img_size=32), id="ape"),
    pytest.param(lambda: _small(in_chans=1), id="gray"),
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
    assert_loads_correctly(HEARTArch(), make)


@pytest.mark.parametrize("make", CASES)
def test_only_heart_claims_heart(make):
    assert _claimants(make().state_dict()) == ["HEART"]


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: HAT(), id="HAT"),
        pytest.param(lambda: DRCT(), id="DRCT"),
        pytest.param(lambda: ATD(), id="ATD"),
        pytest.param(lambda: SwinIR(window_size=8), id="SwinIR"),
        pytest.param(lambda: Swin2SR(window_size=8), id="Swin2SR"),
        pytest.param(lambda: DRFT(), id="DRFT"),
        # SST with RIB attention: the same to_hidden / to_q / to_k parameters
        pytest.param(lambda: sst_arch.sst_light(attn_type="RIB"), id="SST-RIB"),
        pytest.param(lambda: sst_arch.sst_real(), id="SST-Real"),
    ],
)
def test_heart_does_not_claim_relatives(make):
    assert not HEARTArch().detect(make().state_dict())


def test_loaded_model_matches():
    torch.manual_seed(0)
    model = _small(depths=(3, 2), attention_freq=2).eval()
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    loaded = descriptor.model.eval()
    assert descriptor.architecture.id == "HEART"
    assert descriptor.scale == 2

    # not a multiple of the 32 px window, and smaller than one window
    for h, w in ((37, 45), (5, 7)):
        image = torch.rand(1, 3, h, w)
        with torch.no_grad():
            expected = model(image)
            actual = loaded(image)
        assert actual.shape == (1, 3, h * 2, w * 2)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision(dtype):
    torch.manual_seed(0)
    descriptor = MAIN_REGISTRY.load(_small().state_dict())
    assert descriptor.supports_half
    assert descriptor.supports_bfloat16
    model = descriptor.model.eval()
    image = torch.rand(1, 3, 34, 20)
    with torch.no_grad():
        expected = model(image)
        actual = copy.deepcopy(model).to(dtype)(image.to(dtype))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0.05)


def test_size_requirements():
    assert_size_requirements(HEARTArch().load(_small().state_dict()))


def _attention(q, k, v, bias, scale):
    # flash_window.launch's contract with PyTorch's attention: no padding needed
    assert bias is None
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)


def test_inference_attends_through_the_window_kernel(monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(0)
    model = _small(depths=(3, 2)).eval()
    image = torch.rand(1, 3, 37, 45)
    with torch.no_grad():
        expected = model(image)
        calls: list[tuple[int, ...]] = []

        def launch(q, k, v, bias, scale):
            calls.append((*q.shape, v.shape[-1]))
            return _attention(q, k, v, bias, scale)

        monkeypatch.setattr(flash_window, "usable", lambda *_: True)
        monkeypatch.setattr(flash_window, "launch", launch)
        actual = model(image)
    # 2 + 1 attention blocks; q and k carry the 8 position features, unpadded
    assert len(calls) == 3
    assert calls[0][1:] == (2, 1024, 12 + 8, 12)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


def test_training_keeps_pytorchs_attention(monkeypatch: pytest.MonkeyPatch):
    def launch(*_: object) -> torch.Tensor:
        raise AssertionError("the kernel ran")

    monkeypatch.setattr(flash_window, "usable", lambda *_: True)
    monkeypatch.setattr(flash_window, "launch", launch)
    _small().train()(torch.rand(1, 3, 32, 32))


@pytest.mark.skipif(
    not (
        flash_window.AVAILABLE
        and torch.cuda.is_available()
        and torch.version.hip is None
    ),
    reason="the kernel runs on NVIDIA GPUs with Triton",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_window_kernel_matches_pytorchs_attention(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
):
    torch.manual_seed(0)
    model = _small(depths=(3, 2)).eval().cuda().to(dtype)
    image = torch.rand(1, 3, 70, 45, device="cuda", dtype=dtype)
    with torch.no_grad():
        actual = model(image)
        monkeypatch.setattr(flash_window, "FLASH_WINDOW", False)
        expected = model(image)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.02)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_iln_matches_iln(dtype: torch.dtype):
    # eager CUDA inference takes the fused path; its maths runs anywhere
    torch.manual_seed(0)
    norm = heart_arch.iLN(24)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.uniform_(-0.2, 0.2)
    norm = norm.to(dtype)
    x = (torch.randn(2, 37 * 45, 24) * 3 + 1).to(dtype)
    with torch.no_grad():
        expected, expected_std = norm(x)
        actual, std = norm.fused(x)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(std, expected_std, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_exports_need_a_window():
    # the forward's reflect-or-replicate choice is frozen by a trace, so a model is
    # handed no side smaller than its window; called directly it takes any size
    descriptor = HEARTArch().load(_small().state_dict())
    assert descriptor.size_requirements.minimum == 32
    with torch.no_grad():
        assert descriptor(torch.rand(1, 3, 5, 40)).shape == (1, 3, 10, 80)
        assert descriptor.model(torch.rand(1, 3, 5, 40)).shape == (1, 3, 10, 80)
