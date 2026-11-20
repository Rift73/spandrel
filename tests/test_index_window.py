from __future__ import annotations

import pytest
import torch

from spandrel.architectures.__arch_helpers import flash_window, index_window
from spandrel.architectures.HAT import HAT

needs_kernel = pytest.mark.skipif(
    not (
        index_window.AVAILABLE
        and torch.cuda.is_available()
        and torch.version.hip is None
    ),
    reason="the kernel runs on NVIDIA GPUs with Triton",
)


def _hat(window: int = 8) -> HAT:
    torch.manual_seed(0)
    return HAT(
        embed_dim=24,
        depths=(2, 2),
        num_heads=(2, 3),
        window_size=window,
        squeeze_factor=6,
        upscale=2,
    ).eval()


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("b", "h", "w"),
    [
        (1, 40, 56),  # plain, shifted interior and border windows
        (1, 8, 8),  # one window: every shifted window wraps around
        (1, 21, 37),  # padded to the window
        (2, 24, 16),
    ],
)
def test_the_kernel_matches_the_window_copies(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, b: int, h: int, w: int
):
    model = _hat().cuda().to(dtype)
    image = torch.rand(b, 3, h, w, device="cuda", dtype=dtype)
    calls: list[int] = []
    launch = index_window.launch
    monkeypatch.setattr(
        index_window, "launch", lambda *a, **k: calls.append(1) or launch(*a, **k)
    )
    with torch.no_grad():
        actual = model(image)
        # still on: a Triton failure would have switched it off
        assert index_window.INDEX_WINDOW
        assert len(calls) == 4
        monkeypatch.setattr(index_window, "INDEX_WINDOW", False)
        expected = model(image)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.02)


@needs_kernel
@pytest.mark.parametrize("shift", [0, 4])
def test_more_windows_than_a_grid_axis_holds(shift: int):
    # 16,500 windows of 8 x 8 tokens and 4 heads: 66,000 programs, past the 65,535 of a
    # grid's second axis, on one axis; its top rows match a launch under the limit
    torch.manual_seed(0)
    heads, width = 4, 16
    qkv = torch.randn(1, 8 * 132, 8 * 125, 3, heads, width, device="cuda").half()
    bias = torch.randn(heads, 64, 64, device="cuda").half()
    with torch.no_grad():
        whole = index_window.launch(qkv, bias, 8, shift, width, 0.25)
        if shift == 0:
            top = index_window.launch(
                qkv[:, :512].contiguous(), bias, 8, 0, width, 0.25
            )
            assert torch.equal(whole[:, :512], top)
        else:
            assert torch.isfinite(whole).all()


def test_the_kernel_stays_off_on_the_cpu_rocm_and_with_gradients(
    monkeypatch: pytest.MonkeyPatch,
):
    qkv = torch.empty(1, 8, 8, 3, 2, 16, dtype=torch.float16)
    bias = torch.empty(2, 64, 64, dtype=torch.float16)
    with torch.no_grad():
        assert not index_window.usable(qkv, bias, 8)
    monkeypatch.setattr(index_window, "INDEX_WINDOW", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.version, "hip", None)
    with torch.no_grad():
        assert index_window.usable(qkv, bias, 8)
        assert not index_window.usable(qkv.float(), bias.float(), 8)
        monkeypatch.setattr(torch.version, "hip", "7.0")
        assert not index_window.usable(qkv, bias, 8)
    monkeypatch.setattr(torch.version, "hip", None)
    assert not index_window.usable(qkv.requires_grad_(), bias, 8)


class _TritonError(Exception):
    pass


_TritonError.__module__ = "triton.compiler.errors"


def test_a_triton_failure_switches_to_the_window_copies(
    monkeypatch: pytest.MonkeyPatch,
):
    def launch(*_: object, **__: object) -> torch.Tensor:
        raise RuntimeError("kernel launch") from _TritonError("ast.Num")

    monkeypatch.setattr(index_window, "INDEX_WINDOW", True)
    monkeypatch.setattr(index_window, "usable", lambda *_: True)
    monkeypatch.setattr(index_window, "launch", launch)
    qkv = torch.empty(1, 8, 8, 3, 2, 16)
    bias = torch.empty(2, 64, 64)
    assert index_window.index_window_attention(qkv, bias, 8, 0, 16, 1.0) is None
    assert not index_window.INDEX_WINDOW


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("b", "h", "w"), [(1, 48, 80), (2, 32, 48), (1, 16, 16)])
def test_16_px_windows_match_the_window_copies(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, b: int, h: int, w: int
):
    # 256 tokens: four query tiles and four key tiles per window, as released HATs
    model = _hat(16).cuda().to(dtype)
    image = torch.rand(b, 3, h, w, device="cuda", dtype=dtype)
    calls: list[int] = []
    launch = index_window.launch
    monkeypatch.setattr(
        index_window, "launch", lambda *a, **k: calls.append(1) or launch(*a, **k)
    )
    with torch.no_grad():
        actual = model(image)
        assert index_window.INDEX_WINDOW
        assert len(calls) == 4
        monkeypatch.setattr(index_window, "INDEX_WINDOW", False)
        expected = model(image)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.02)


@needs_kernel
@pytest.mark.parametrize("window", [12, 7])
def test_the_kernel_declines_windows_of_part_tiles(window: int):
    qkv = torch.zeros(1, 2 * window, 3 * window, 3, 2, 16, device="cuda").half()
    bias = torch.zeros(2, window**2, window**2, device="cuda").half()
    with torch.no_grad():
        assert not index_window.usable(qkv, bias, window)
        assert (
            index_window.index_window_attention(qkv, bias, window, 0, 16, 1.0) is None
        )
        with pytest.raises(ValueError):
            index_window.launch(qkv, bias, window, 0, 16, 1.0)
    assert index_window.INDEX_WINDOW


@needs_kernel
def test_12_px_windows_keep_the_window_copies(monkeypatch: pytest.MonkeyPatch):
    # 144 tokens: not whole 64-token tiles; the model runs as it does without the kernel
    window = 12
    model = _hat(window).cuda().half()
    image = torch.rand(1, 3, 2 * window, 3 * window, device="cuda").half()
    calls: list[int] = []
    launch = index_window.launch
    monkeypatch.setattr(
        index_window, "launch", lambda *a, **k: calls.append(1) or launch(*a, **k)
    )
    with torch.no_grad():
        actual = model(image)
        assert index_window.INDEX_WINDOW
        assert calls == []
        monkeypatch.setattr(index_window, "INDEX_WINDOW", False)
        expected = model(image)
    assert torch.equal(actual, expected)


@needs_kernel
def test_shifted_windows_past_a_grid_axis_match_the_window_copies(
    monkeypatch: pytest.MonkeyPatch,
):
    # 8 px windows of 3 heads over a 1184 x 1184 trunk: 65,712 windows * heads, the
    # one-axis launch, shifted blocks included; the copies' reference runs on PyTorch's
    # attention (flash_window off), which takes any number of windows
    torch.manual_seed(0)
    model = (
        HAT(
            embed_dim=24,
            depths=(2,),
            num_heads=(3,),
            window_size=8,
            squeeze_factor=6,
            upscale=1,
        )
        .eval()
        .cuda()
        .half()
    )
    image = torch.rand(1, 3, 1184, 1184, device="cuda").half()
    calls: list[int] = []
    launch = index_window.launch
    monkeypatch.setattr(
        index_window, "launch", lambda *a, **k: calls.append(1) or launch(*a, **k)
    )
    # the overlapping attention keeps flash_window in both runs, off: it is not under
    # test here
    monkeypatch.setattr(flash_window, "FLASH_WINDOW", False)
    with torch.no_grad():
        actual = model(image)
        assert index_window.INDEX_WINDOW
        assert len(calls) == 2
        monkeypatch.setattr(index_window, "INDEX_WINDOW", False)
        expected = model(image)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0.02)


def _rolled_windows(qkv, bias, window, shift, width, scale):
    """The attention of the image's windows rolled by (-shift, -shift), without a mask,
    in FP32 PyTorch operations, as (B, H, W, heads * width)."""
    b, h, w, _, heads, _ = qkv.shape
    x = torch.roll(qkv.float(), (-shift, -shift), (1, 2))[..., :width]
    x = x.view(b, h // window, window, w // window, window, 3, heads, width)
    x = x.permute(5, 0, 1, 3, 6, 2, 4, 7).reshape(3, -1, heads, window * window, width)
    q, k, v = x.unbind(0)
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale + bias.float(), -1) @ v
    y = attn.view(b, h // window, w // window, heads, window, window, width)
    y = y.permute(0, 1, 4, 2, 5, 3, 6).reshape(b, h, w, heads * width)
    return torch.roll(y, (shift, shift), (1, 2))


@needs_kernel
@pytest.mark.parametrize(("window", "shift"), [(8, 4), (16, 8), (8, 0)])
def test_shifted_windows_without_the_mask(window: int, shift: int):
    # TFDATv2's shifted windows wrap around unmasked; masked=True (the default) is
    # Swin's, which differs exactly in the wrapped windows
    generator = torch.Generator().manual_seed(4)
    qkv = torch.randn(2, 2 * window, 3 * window, 3, 2, 16, generator=generator)
    bias = torch.randn(2, window * window, window * window, generator=generator)
    qkv, bias = qkv.cuda().half(), bias.cuda().half()
    scale = 16**-0.5
    unmasked = index_window.launch(qkv, bias, window, shift, 16, scale, masked=False)
    expected = _rolled_windows(qkv, bias, window, shift, 16, scale)
    torch.testing.assert_close(unmasked.float(), expected, rtol=0, atol=2e-3)
    masked = index_window.launch(qkv, bias, window, shift, 16, scale)
    assert torch.equal(masked, unmasked) == (shift == 0)
