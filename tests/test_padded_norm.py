from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.nn.functional import layer_norm, pad

from spandrel.architectures.__arch_helpers import padded_norm

needs_kernel = pytest.mark.skipif(
    not (
        padded_norm.AVAILABLE
        and torch.cuda.is_available()
        and torch.version.hip is None
    ),
    reason="the kernel runs on NVIDIA GPUs with Triton",
)


def norm(real: int = 180, padded: int = 184) -> padded_norm.PaddedLayerNorm:
    torch.manual_seed(0)
    source = nn.LayerNorm(real)
    nn.init.normal_(source.weight, 1, 0.2)
    nn.init.normal_(source.bias, 0, 0.2)
    return padded_norm.PaddedLayerNorm(source, padded).eval()


def padded_tokens(
    rows: int, real: int, padded: int, dtype: torch.dtype
) -> torch.Tensor:
    """A trunk's tokens: real channels, then exact zeros."""
    x = torch.randn(rows, real, device="cuda") * 3 + 0.5
    return pad(x, (0, padded - real)).to(dtype)


def check(module: padded_norm.PaddedLayerNorm, x: torch.Tensor) -> None:
    real = module.real
    with torch.no_grad():
        out = module(x)
    reference = layer_norm(
        x[..., :real].float(), (real,), module.weight.float(), module.bias.float(), 1e-5
    )
    expected = layer_norm(
        x[..., :real], (real,), module.weight.to(x.dtype), module.bias.to(x.dtype), 1e-5
    )
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.equal(out[..., real:], torch.zeros_like(out[..., real:]))
    # Within half-precision rounding: no further from FP32 than F.layer_norm's own.
    kernel_error = (out[..., :real].float() - reference).abs().max().item()
    torch_error = (expected.float() - reference).abs().max().item()
    assert kernel_error <= 2 * torch_error + 1e-3, (kernel_error, torch_error)


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("real", "padded"), [(180, 184), (60, 64), (180, 180)])
def test_the_kernel_matches_layer_norm_on_the_real_channels(
    real: int, padded: int, dtype: torch.dtype
):
    module = norm(real, padded).cuda().to(dtype)
    x = padded_tokens(4099, real, padded, dtype)
    with torch.no_grad():
        assert padded_norm.usable(x, module)
    check(module, x)


@needs_kernel
def test_strided_tokens_read_in_place():
    # A slice of a wider tensor: rows 192 apart, 184 of them read.
    module = norm().cuda().half()
    wide = pad(padded_tokens(1000, 180, 184, torch.float16), (0, 8))
    x = wide[:, :184]
    assert not x.is_contiguous()
    check(module, x.view(10, 100, 184))


@needs_kernel
def test_an_fp32_sum_with_fp16_weights_comes_back_in_fp16():
    # HAT's and DAT's last group hands its norms an FP32 sum (fp32_last_group).
    module = norm().cuda().half()
    x = padded_tokens(1000, 180, 184, torch.float32) * 1000
    with torch.no_grad():
        assert padded_norm.usable(x, module)
        out = module(x)
    assert out.dtype == torch.float16
    reference = layer_norm(
        x[:, :180], (180,), module.weight.float(), module.bias.float(), 1e-5
    )
    torch.testing.assert_close(out[:, :180].float(), reference, atol=1e-2, rtol=1e-3)
    assert torch.equal(out[:, 180:], torch.zeros_like(out[:, 180:]))


def test_the_pytorch_path_computes_the_same_values():
    module = norm()
    x = pad(torch.randn(7, 180), (0, 4))
    with torch.no_grad():
        out = module(x)
    assert torch.equal(out[:, 180:], torch.zeros(7, 4))
    torch.testing.assert_close(
        out[:, :180], layer_norm(x[:, :180], (180,), module.weight, module.bias)
    )


def test_the_kernel_stays_off_on_rocm_builds(monkeypatch: pytest.MonkeyPatch):
    module = norm().half()
    x = torch.empty(4, 184, dtype=torch.float16)
    monkeypatch.setattr(padded_norm, "PADDED_NORM", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.version, "hip", None)
    with torch.no_grad():
        assert padded_norm.usable(x, module)
        monkeypatch.setattr(torch.version, "hip", "7.0")
        assert not padded_norm.usable(x, module)


def test_fp32_weights_and_gradients_keep_pytorchs_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(padded_norm, "PADDED_NORM", True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.version, "hip", None)
    with torch.no_grad():
        assert not padded_norm.usable(torch.empty(4, 184), norm())
    assert not padded_norm.usable(
        torch.empty(4, 184, dtype=torch.float16, requires_grad=True), norm().half()
    )


class _TritonError(Exception):
    pass


_TritonError.__module__ = "triton.compiler.errors"


def test_a_triton_failure_switches_to_the_pytorch_path(monkeypatch: pytest.MonkeyPatch):
    def launch(*_: object) -> torch.Tensor:
        raise RuntimeError("kernel launch") from _TritonError("ast.Num")

    module = norm()
    monkeypatch.setattr(padded_norm, "PADDED_NORM", True)
    monkeypatch.setattr(padded_norm, "usable", lambda *_: True)
    monkeypatch.setattr(padded_norm, "launch", launch)
    x = pad(torch.randn(3, 180), (0, 4))
    with torch.no_grad():
        out = module(x)
    assert not padded_norm.PADDED_NORM
    assert torch.equal(out[:, 180:], torch.zeros(3, 4))


def test_other_failures_are_raised(monkeypatch: pytest.MonkeyPatch):
    def launch(*_: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("CUDA out of memory")

    module = norm()
    monkeypatch.setattr(padded_norm, "PADDED_NORM", True)
    monkeypatch.setattr(padded_norm, "usable", lambda *_: True)
    monkeypatch.setattr(padded_norm, "launch", launch)
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError):
        module(torch.zeros(3, 184))
    assert padded_norm.PADDED_NORM


def test_the_source_must_be_affine_and_narrower():
    with pytest.raises(ValueError, match="affine"):
        padded_norm.PaddedLayerNorm(nn.LayerNorm(180, elementwise_affine=False), 184)
    with pytest.raises(ValueError, match="affine"):
        padded_norm.PaddedLayerNorm(nn.LayerNorm(180), 176)


def kernel_norm(width: int = 180) -> padded_norm.KernelLayerNorm:
    torch.manual_seed(0)
    module = padded_norm.KernelLayerNorm(width)
    nn.init.normal_(module.weight, 1, 0.2)
    nn.init.normal_(module.bias, 0, 0.2)
    return module.eval()


@needs_kernel
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [180, 192, 144])
def test_the_unpadded_kernel_matches_layer_norm(width: int, dtype: torch.dtype):
    module = kernel_norm(width).cuda().to(dtype)
    x = (torch.randn(3, 1001, width, device="cuda") * 3 + 0.5).to(dtype)
    launches: list[int] = []
    launch = padded_norm.launch

    def counted(*args: Any, **kwargs: Any) -> torch.Tensor:
        launches.append(1)
        return launch(*args, **kwargs)

    with torch.no_grad():
        padded_norm.launch = counted
        try:
            out = module(x)
        finally:
            padded_norm.launch = launch
        expected = nn.LayerNorm.forward(module, x)
    reference = layer_norm(
        x.float(), (width,), module.weight.float(), module.bias.float(), 1e-5
    )
    assert launches and out.dtype == dtype and out.shape == x.shape
    kernel_error = (out.float() - reference).abs().max().item()
    torch_error = (expected.float() - reference).abs().max().item()
    assert kernel_error <= 2 * torch_error + 1e-3, (kernel_error, torch_error)


def test_the_unpadded_norm_is_a_layer_norm_with_its_keys():
    module = kernel_norm()
    assert isinstance(module, nn.LayerNorm)
    assert set(module.state_dict()) == {"weight", "bias"}
    x = torch.randn(5, 180)
    with torch.no_grad():
        torch.testing.assert_close(module(x), nn.LayerNorm.forward(module, x))


def test_the_unpadded_norm_takes_pytorchs_path_where_the_kernel_may_not_run(
    monkeypatch: pytest.MonkeyPatch,
):
    def launch(*_: object) -> torch.Tensor:
        raise AssertionError("the kernel ran")

    monkeypatch.setattr(padded_norm, "launch", launch)
    module = kernel_norm()
    with torch.no_grad():
        module(torch.randn(5, 180))  # the CPU; FP32
    monkeypatch.setattr(padded_norm, "usable", lambda *_: True)
    # An input in another type than the weights: nn.LayerNorm's own error, not the kernel.
    with torch.no_grad(), pytest.raises(RuntimeError, match="dtype"):
        module.half()(torch.randn(5, 180))


def test_a_triton_failure_switches_the_unpadded_norm_to_pytorch(
    monkeypatch: pytest.MonkeyPatch,
):
    def launch(*_: object) -> torch.Tensor:
        raise RuntimeError("kernel launch") from _TritonError("ast.Num")

    monkeypatch.setattr(padded_norm, "PADDED_NORM", True)
    monkeypatch.setattr(padded_norm, "usable", lambda *_: True)
    monkeypatch.setattr(padded_norm, "launch", launch)
    module = kernel_norm()
    x = torch.randn(5, 180)
    with torch.no_grad():
        torch.testing.assert_close(module(x), nn.LayerNorm.forward(module, x))
    assert not padded_norm.PADDED_NORM
