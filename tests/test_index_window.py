from __future__ import annotations

import pytest
import torch

from spandrel.architectures.__arch_helpers import index_window
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
