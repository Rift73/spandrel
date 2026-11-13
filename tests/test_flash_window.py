from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from spandrel.architectures.__arch_helpers import flash_window

needs_kernel = pytest.mark.skipif(
    not (
        flash_window.AVAILABLE
        and torch.cuda.is_available()
        and torch.version.hip is None
    ),
    reason="the kernel runs on NVIDIA GPUs with Triton",
)


def relative_position_index(window_h: int, window_w: int) -> torch.Tensor:
    """(N, N) as HAT's and DAT's attention build it."""
    coords = torch.stack(
        torch.meshgrid(torch.arange(window_h), torch.arange(window_w), indexing="ij")
    ).flatten(1)
    relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window_h - 1
    relative[:, :, 1] += window_w - 1
    relative[:, :, 0] *= 2 * window_w - 1
    return relative.sum(-1)


def errors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    table: torch.Tensor | None,
    window: tuple[int, int],
) -> tuple[float, float]:
    """The kernel's and scaled_dot_product_attention's (the bias as a float mask, as
    the architectures pass it) largest error against FP32 attention."""
    mask = None
    if table is not None:
        mask = table[:, relative_position_index(*window).to(table.device)].unsqueeze(0)
    reference = scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=None if mask is None else mask.float(),
        scale=scale,
    )
    sdpa = scaled_dot_product_attention(
        q, k, v, attn_mask=None if mask is None else mask.to(q.dtype), scale=scale
    )
    kernel = flash_window.launch(q, k, v, scale, table, window[1])
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


# windows, heads, window (h, w), head width: HAT-L's (16 x 16 windows of 6 heads), its
# heads unpadded, and DAT2's (3 heads; a rectangular window of its other branch)
BIASED = {
    "HAT-L": (64, 6, (16, 16), 32),
    "HAT-L, unpadded heads": (64, 6, (16, 16), 30),
    "DAT2": (64, 3, (16, 16), 32),
    "DAT2, rectangular": (64, 3, (8, 32), 32),
}


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", BIASED)
def test_biased_windows_match_attention_with_the_mask(case: str, dtype: torch.dtype):
    windows, heads, window, width = BIASED[case]
    q, k, v = operands(windows, heads, window[0] * window[1], width, width, dtype)
    size = (2 * window[0] - 1) * (2 * window[1] - 1)
    table = torch.randn(heads, size, device="cuda").to(dtype)
    kernel, sdpa = errors(q, k, v, width**-0.5, table, window)
    # Within half-precision rounding: no further from FP32 than PyTorch's own kernel.
    assert kernel <= 2 * sdpa + 1e-3, (kernel, sdpa)


# DRFT's unmasked windows: its rank-32 bias rides in q and k (head 32 + rank 32), v
# keeps the head's width; windows of 16 and 32.
@needs_kernel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [16, 32])
def test_unbiased_windows_match_attention(window: int, dtype: torch.dtype):
    q, k, v = operands(32, 6, window * window, 64, 32, dtype)
    kernel, sdpa = errors(q, k, v, 32**-0.5, None, (window, window))
    assert kernel <= 2 * sdpa + 1e-3, (kernel, sdpa)


@needs_kernel
def test_strided_windows_read_in_place():
    # img2windows' views: q, k and v are slices of one tensor, not copies.
    q, k, v = operands(16, 6, 256, 32, 32, torch.float16)
    qkv = torch.stack([q, k, v], dim=3)
    q_view, k_view, v_view = qkv.unbind(3)
    assert not q_view.is_contiguous()
    assert torch.equal(
        flash_window.launch(q_view, k_view, v_view, 32**-0.5, None, 16),
        flash_window.launch(q, k, v, 32**-0.5, None, 16),
    )


def test_the_kernel_stays_off_on_rocm_builds(monkeypatch: pytest.MonkeyPatch):
    module = nn.Identity().eval()
    q = torch.empty(1, 1, 16, 32, dtype=torch.float16)
    monkeypatch.setattr(flash_window, "FLASH_WINDOW", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.version, "hip", None)
    with torch.no_grad():
        assert flash_window.usable(q, q, q, module)
        monkeypatch.setattr(torch.version, "hip", "7.0")
        assert not flash_window.usable(q, q, q, module)


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
    assert flash_window.flash_window_attention(q, q, q, 1.0, nn.Identity()) is None
    assert not flash_window.FLASH_WINDOW


def test_other_failures_are_raised(monkeypatch: pytest.MonkeyPatch):
    def launch(*_: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(flash_window, "FLASH_WINDOW", True)
    monkeypatch.setattr(flash_window, "usable", lambda *_: True)
    monkeypatch.setattr(flash_window, "launch", launch)
    q = torch.empty(1, 1, 16, 32)
    with pytest.raises(torch.OutOfMemoryError):
        flash_window.flash_window_attention(q, q, q, 1.0, nn.Identity())
    assert flash_window.FLASH_WINDOW
