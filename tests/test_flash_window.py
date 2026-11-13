from __future__ import annotations

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from spandrel.architectures.__arch_helpers import flash_window
from spandrel.architectures.__arch_helpers.attention import window_attention

needs_kernel = pytest.mark.skipif(
    not (
        flash_window.AVAILABLE
        and torch.cuda.is_available()
        and torch.version.hip is None
    ),
    reason="the kernel runs on NVIDIA GPUs with Triton",
)


def errors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
) -> tuple[float, float]:
    """The kernel's and scaled_dot_product_attention's (the bias as its float mask, as
    window_attention passes it) largest error against FP32 attention."""
    reference = scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=None if bias is None else bias.float(),
        scale=scale,
    )
    sdpa = scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
    kernel = flash_window.launch(q, k, v, bias, scale)
    assert kernel.shape == sdpa.shape
    assert torch.isfinite(kernel).all()
    return (
        (kernel.float() - reference).abs().max().item(),
        (sdpa.float() - reference).abs().max().item(),
    )


def operands(
    windows: int, heads: int, tokens: int, d_qk: int, d_v: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    shape = (windows, heads, tokens)
    return (
        torch.randn(*shape, d_qk, device="cuda").to(dtype),
        torch.randn(*shape, d_qk, device="cuda").to(dtype),
        torch.randn(*shape, d_v, device="cuda").to(dtype),
    )


# windows, heads, tokens, head width: HAT-L's (16 x 16 windows of 6 heads), its heads
# unpadded, and DAT2's (3 heads)
BIASED = {
    "HAT-L": (64, 6, 256, 32),
    "HAT-L, unpadded heads": (64, 6, 256, 30),
    "DAT2": (64, 3, 256, 32),
}


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", BIASED)
def test_biased_windows_match_attention_with_the_mask(case: str, dtype: torch.dtype):
    windows, heads, tokens, width = BIASED[case]
    q, k, v = operands(windows, heads, tokens, width, width, dtype)
    bias = torch.randn(1, heads, tokens, tokens, device="cuda").to(dtype)
    kernel, sdpa = errors(q, k, v, bias, width**-0.5)
    # Within half-precision rounding: no further from FP32 than PyTorch's own kernel.
    assert kernel <= 2 * sdpa + 1e-3, (kernel, sdpa)


# DRFT's unmasked windows: its rank-32 bias rides in q and k (head 32 + rank 32), v
# keeps the head's width; windows of 16 and 32.
@needs_kernel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [16, 32])
def test_unbiased_windows_match_attention(window: int, dtype: torch.dtype):
    q, k, v = operands(32, 6, window * window, 64, 32, dtype)
    kernel, sdpa = errors(q, k, v, None, 1.0)
    assert kernel <= 2 * sdpa + 1e-3, (kernel, sdpa)


@needs_kernel
def test_strided_windows_read_in_place():
    # img2windows' views: q, k and v are slices of one tensor, not copies.
    q, k, v = operands(16, 6, 256, 32, 32, torch.float16)
    qkv = torch.stack([q, k, v], dim=3)
    q_view, k_view, v_view = qkv.unbind(3)
    assert not q_view.is_contiguous()
    assert torch.equal(
        flash_window.launch(q_view, k_view, v_view, None, 32**-0.5),
        flash_window.launch(q, k, v, None, 32**-0.5),
    )


@needs_kernel
def test_window_attention_runs_unmasked_windows_in_the_kernel():
    q, k, v = operands(16, 6, 256, 32, 32, torch.float16)
    bias = torch.randn(1, 6, 256, 256, device="cuda").half()
    with torch.no_grad():
        attended = window_attention(q, k, v, bias, None, 32**-0.5)
    assert torch.equal(attended, flash_window.launch(q, k, v, bias, 32**-0.5))


def test_masked_windows_keep_pytorchs_attention(monkeypatch: pytest.MonkeyPatch):
    def launch(*_: object) -> torch.Tensor:
        raise AssertionError("the kernel ran")

    monkeypatch.setattr(flash_window, "usable", lambda *_: True)
    monkeypatch.setattr(flash_window, "launch", launch)
    q = torch.rand(4, 2, 16, 8)
    with torch.no_grad():
        window_attention(q, q, q, None, torch.zeros(4, 16, 16), 0.5)


def test_gradients_keep_pytorchs_attention(monkeypatch: pytest.MonkeyPatch):
    q = torch.empty(1, 1, 16, 32, dtype=torch.float16, requires_grad=True)
    monkeypatch.setattr(flash_window, "FLASH_WINDOW", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.version, "hip", None)
    assert not flash_window.usable(q, q, q, None)
    with torch.no_grad():
        assert flash_window.usable(q.detach(), q, q, None)


def test_the_kernel_stays_off_on_rocm_builds(monkeypatch: pytest.MonkeyPatch):
    q = torch.empty(1, 1, 16, 32, dtype=torch.float16)
    monkeypatch.setattr(flash_window, "FLASH_WINDOW", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.version, "hip", None)
    with torch.no_grad():
        assert flash_window.usable(q, q, q, None)
        monkeypatch.setattr(torch.version, "hip", "7.0")
        assert not flash_window.usable(q, q, q, None)


class _TritonError(Exception):
    pass


_TritonError.__module__ = "triton.compiler.errors"


def test_a_triton_failure_switches_to_the_pytorch_path(monkeypatch: pytest.MonkeyPatch):
    def launch(*_: object) -> torch.Tensor:
        raise RuntimeError("kernel launch") from _TritonError("ast.Num")

    monkeypatch.setattr(flash_window, "FLASH_WINDOW", True)
    monkeypatch.setattr(flash_window, "usable", lambda *_: True)
    monkeypatch.setattr(flash_window, "launch", launch)
    q = torch.empty(1, 1, 16, 32)
    assert flash_window.flash_window_attention(q, q, q, None, 1.0) is None
    assert not flash_window.FLASH_WINDOW


def test_other_failures_are_raised(monkeypatch: pytest.MonkeyPatch):
    def launch(*_: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(flash_window, "FLASH_WINDOW", True)
    monkeypatch.setattr(flash_window, "usable", lambda *_: True)
    monkeypatch.setattr(flash_window, "launch", launch)
    q = torch.empty(1, 1, 16, 32)
    with pytest.raises(torch.OutOfMemoryError):
        flash_window.flash_window_attention(q, q, q, None, 1.0)
    assert flash_window.FLASH_WINDOW
