from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from einops import rearrange
from torch import nn

from spandrel import MAIN_REGISTRY
from spandrel.architectures.__arch_helpers import flash_window
from spandrel.architectures.__arch_helpers.inference import tail_scale_and_halo
from spandrel.architectures.ATD import ATD
from spandrel.architectures.DRCT import DRCT
from spandrel.architectures.DRFT import DRFT
from spandrel.architectures.HAT import HAT
from spandrel.architectures.HEART import HEART, HEARTArch
from spandrel.architectures.HEART.__arch import heart_arch, iln_triton
from spandrel.architectures.HEART.__arch import window_triton as heart_window
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


def _forced_bands(monkeypatch: pytest.MonkeyPatch, rows: int) -> None:
    """The banded tail on the CPU, in bands of `rows` trunk rows."""
    monkeypatch.setattr(
        HEART,
        "tail_plan",
        lambda self, _x: tail_scale_and_halo(
            nn.Sequential(self.conv_before_upsample, self.upsample, self.conv_last)
        ),
    )
    monkeypatch.setattr(heart_arch, "WHOLE_TAIL_ROWS", 0)
    monkeypatch.setattr(heart_arch, "MIN_BAND_ROWS", rows)
    monkeypatch.setattr(heart_arch, "BAND_DIVISOR", 10**6)


@pytest.mark.parametrize("scale", [2, 3, 4, 8])
@pytest.mark.parametrize(("h", "w"), [(37, 45), (5, 7), (64, 9)])
def test_banded_tail_computes_the_whole_pass(
    monkeypatch: pytest.MonkeyPatch, scale: int, h: int, w: int
):
    torch.manual_seed(0)
    model = _small(upscale=scale).eval()
    image = torch.rand(1, 3, h, w)
    with torch.no_grad():
        whole = model(image)
        # 7 rows: the padded trunk (32 or 64 rows) is not a whole number of bands
        _forced_bands(monkeypatch, rows=7)
        banded = model(image)
    assert banded.shape == whole.shape == (1, 3, h * scale, w * scale)
    # a band's convolutions may sum in another order: FP32 rounding
    torch.testing.assert_close(banded, whole, atol=1e-5, rtol=0)


def test_an_image_shorter_than_one_band_runs_the_tail_whole(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _small(upscale=4).eval()
    _forced_bands(monkeypatch, rows=64)
    calls: list[int] = []
    tail = model.tail
    monkeypatch.setattr(model, "tail", lambda t: calls.append(t.shape[2]) or tail(t))
    with torch.no_grad():
        model(torch.rand(1, 3, 20, 20))
    assert calls == [32]


def test_tail_plan_bands_only_eager_nvidia_inference(monkeypatch: pytest.MonkeyPatch):
    x = torch.rand(1, 24, 8, 8)
    with torch.no_grad():
        # the CPU keeps the whole pass
        assert _small(upscale=4).eval().tail_plan(x) is None
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
        monkeypatch.setattr(torch.version, "hip", None)
        assert _small(upscale=4).eval().tail_plan(x) == (4, 4)
        assert _small(upscale=2).eval().tail_plan(x) == (2, 3)
        # nothing to band at 1x
        assert _small(upscale=1).eval().tail_plan(x) is None
        monkeypatch.setattr(torch.version, "hip", "7.0")
        assert _small(upscale=4).eval().tail_plan(x) is None


def _attention_before_early_freeing(attn, x: torch.Tensor) -> torch.Tensor:
    """RIBWindowAttention as published: one rearrange of q, k and v together, kept to the
    end."""
    _, _, h, w = x.shape
    ws = attn.window_size[0]
    s = ws // 2 if attn.shift else 0
    if s:
        x = torch.nn.functional.pad(x, (s, s, s, s), mode="reflect")
    hd, wd = x.shape[2] // ws, x.shape[3] // ws
    qkv = rearrange(
        attn.to_qkv(x),
        "b (qkv heads c) (h wh) (w ww) -> qkv (b h w) (wh ww) heads c",
        heads=attn.num_heads,
        wh=ws,
        ww=ws,
        qkv=3,
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    hidden = torch.relu(attn.rib_coords @ attn.to_hidden + attn.hidden_b)
    q_pos = torch.einsum("nd,hdr->hnr", hidden, attn.to_q).expand(
        q.shape[0], -1, -1, -1
    )
    k_pos = torch.einsum("nd,hdr->hnr", hidden, attn.to_k).expand(
        q.shape[0], -1, -1, -1
    )
    d, r = q.shape[-1], q_pos.shape[-1]
    q_cat = torch.cat([q * d**-0.5, q_pos.transpose(1, 2) * r**-0.5], -1)
    k_cat = torch.cat([k, k_pos.transpose(1, 2)], -1)
    pad = -q_cat.shape[-1] % 8
    out = torch.nn.functional.scaled_dot_product_attention(
        torch.nn.functional.pad(q_cat, (0, pad)).transpose(1, 2),
        torch.nn.functional.pad(k_cat, (0, pad)).transpose(1, 2),
        torch.nn.functional.pad(v, (0, r + pad)).transpose(1, 2),
        scale=1.0,
    ).transpose(1, 2)[..., :d]
    out = rearrange(
        out,
        "(b h w) (wh ww) heads c -> b (heads c) (h wh) (w ww)",
        h=hd,
        w=wd,
        wh=ws,
        ww=ws,
    )
    return attn.to_out(out[:, :, s : s + h, s : s + w])


@pytest.mark.parametrize("shift", [False, True])
def test_early_freeing_keeps_the_attentions_output(shift: bool):
    torch.manual_seed(0)
    attn = heart_arch.RIBWindowAttention(
        dim=24, window_size=32, num_heads=2, shift=shift
    )
    for p in attn.parameters():
        nn.init.normal_(p, 0, 0.1)
    attn.eval()
    x = torch.rand(1, 24, 64, 96)
    with torch.no_grad():
        assert torch.equal(attn(x), _attention_before_early_freeing(attn, x))


def _forced_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """The token layout on the CPU, for inference."""
    monkeypatch.setattr(
        heart_arch,
        "token_layout",
        lambda _x, module: not module.training and not torch.is_grad_enabled(),
    )


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _small(depths=(3, 2)), id="shifted-and-plain"),
        pytest.param(lambda: _small(attention_freq=1, upscale=4), id="every-block"),
        pytest.param(lambda: _small(resi_connection="identity"), id="identity"),
        pytest.param(lambda: _small(upscale=1, patch_norm=False), id="1x"),
    ],
)
@pytest.mark.parametrize(("h", "w"), [(37, 45), (5, 7), (64, 33)])
def test_token_layout_computes_the_same_model(
    monkeypatch: pytest.MonkeyPatch, make, h: int, w: int
):
    torch.manual_seed(0)
    model = make().eval()
    for p in model.parameters():
        nn.init.normal_(p, 0, 0.05)
    image = torch.rand(1, 3, h, w)
    with torch.no_grad():
        expected = model(image)
        _forced_tokens(monkeypatch)
        actual = model(image)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_token_layout_stays_off_where_the_original_layout_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _small().eval()
    x = torch.rand(1, 64, 24)
    with torch.no_grad():
        # the CPU
        assert not heart_arch.token_layout(x, model)
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
        monkeypatch.setattr(torch.version, "hip", None)
        assert heart_arch.token_layout(x, model)
        # ROCm
        monkeypatch.setattr(torch.version, "hip", "7.0")
        assert not heart_arch.token_layout(x, model)
        monkeypatch.setattr(torch.version, "hip", None)
    # gradients and training
    assert not heart_arch.token_layout(x, model)
    with torch.no_grad():
        assert not heart_arch.token_layout(x, model.train())


@pytest.mark.skipif(
    not (
        heart_window.AVAILABLE
        and torch.cuda.is_available()
        and torch.version.hip is None
    ),
    reason="the kernel runs on NVIDIA GPUs with Triton",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("h", "w"), [(70, 45), (32, 32), (20, 100)])
def test_heart_window_kernel_matches_the_gathered_windows(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, h: int, w: int
):
    torch.manual_seed(0)
    model = _small(depths=(3, 2)).eval().cuda().to(dtype)
    image = torch.rand(1, 3, h, w, device="cuda", dtype=dtype)
    with torch.no_grad():
        actual = model(image)
        monkeypatch.setattr(heart_window, "HEART_WINDOW", False)
        expected = model(image)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.02)


def test_the_heart_window_kernel_stays_off_on_the_cpu():
    qkv = torch.rand(1, 32, 32, 72).half()
    with torch.no_grad():
        assert not heart_window.usable(qkv, 32)


@pytest.mark.parametrize(("real", "padded"), [(24, 32), (180, 184), (24, 24)])
def test_padded_iln_counts_the_real_channels(real: int, padded: int):
    # the CPU computes the kernels' values with PyTorch's operations
    torch.manual_seed(0)
    source = heart_arch.iLN(real)
    with torch.no_grad():
        source.weight.uniform_(0.5, 1.5)
        source.bias.uniform_(-0.2, 0.2)
    norm = iln_triton.PaddedILN(source, padded).eval()
    x = torch.randn(2, 37 * 45, real) * 3 + 1
    with torch.no_grad():
        expected, expected_std = source(x)
        actual, std = norm(torch.nn.functional.pad(x, (0, padded - real)))
    torch.testing.assert_close(std, expected_std, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[..., :real], expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(actual[..., real:], torch.zeros_like(actual[..., real:]))
    with pytest.raises(ValueError):
        iln_triton.PaddedILN(source, real - 1)
