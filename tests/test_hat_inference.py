from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch
from torch import nn

from spandrel import ModelLoader
from spandrel.architectures.__arch_helpers.inference import (
    border_window_mask,
    shifted_window_mask,
)
from spandrel.architectures.ESRGAN import ESRGAN
from spandrel.architectures.HAT import HAT
from spandrel.architectures.HAT.__arch import HAT as HAT_MODULE


def tiny(**settings) -> nn.Module:
    return HAT(
        embed_dim=settings.pop("embed_dim", 30),
        depths=settings.pop("depths", (3, 3)),
        num_heads=settings.pop("num_heads", (2, 2)),
        window_size=8,
        compress_ratio=2,
        squeeze_factor=4,
        upsampler="pixelshuffle",
        upscale=2,
        **settings,
    )


MODELS: dict[str, Callable[[], nn.Module]] = {
    "padded heads": tiny,
    "heads of a multiple of 8": lambda: tiny(embed_dim=48, num_heads=(3, 3)),
    "qkv bias off": lambda: tiny(qkv_bias=False),
    "no closing convolution": lambda: tiny(resi_connection="identity"),
}
# Multiples of the windows and not, wider than tall and taller than wide, one window's size
SIZES = [(16, 16), (24, 40), (40, 24), (21, 37), (8, 8), (8, 32), (32, 8)]


def model(name: str) -> nn.Module:
    torch.manual_seed(0)
    return ModelLoader().load_from_state_dict(MODELS[name]().state_dict()).model.eval()


def image(height: int, width: int, batch: int = 1) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(batch, 3, height, width)


def traced_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The path an ONNX export traces, with nothing kept."""
    monkeypatch.setattr(HAT_MODULE, "traced", lambda: True)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("size", SIZES, ids=str)
@pytest.mark.parametrize("batch", [1, 2])
def test_inference_computes_what_the_traced_path_computes(
    monkeypatch: pytest.MonkeyPatch, name: str, size: tuple[int, int], batch: int
):
    net = model(name)
    x = image(*size, batch)
    with torch.no_grad():
        actual = net(x)
        again = net(x)
        with monkeypatch.context() as patch:
            traced_path(patch)
            expected = net(x)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
    assert torch.equal(actual, again)
    assert actual.is_contiguous()


def test_gradients_take_the_same_values_and_keep_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    net = model("padded heads")
    x = image(16, 16)
    actual = net(x)
    with torch.no_grad(), monkeypatch.context() as patch:
        traced_path(patch)
        expected = net(x)
    torch.testing.assert_close(actual.detach(), expected, atol=2e-5, rtol=0)
    actual.sum().backward()
    assert not any(getattr(m, "derived", None) for m in net.modules())
    assert not getattr(net, "window_masks", None)


def test_a_run_keeps_one_mask_for_all_blocks_and_only_the_border_windows():
    net = model("padded heads")
    with torch.no_grad():
        net(image(32, 48))
    assert len(net.window_masks) == 1
    mask = next(iter(net.window_masks.values()))
    full = shifted_window_mask(32, 48, 8, 8, 4, 4, torch.float32, torch.device("cpu"))
    assert mask.shape[0] < full.shape[0]
    marked = full.flatten(1).ne(0).any(1)
    assert marked.sum() == mask.shape[0]
    border = border_window_mask(32, 48, 8, 8, 4, 4, torch.float32, torch.device("cpu"))
    assert torch.equal(mask, border)


def test_what_inference_keeps_follows_new_weights():
    net = model("padded heads")
    x = image(16, 16)
    with torch.no_grad():
        net(x)
    other = model("padded heads")
    state = {
        k: v + 0.01 if v.is_floating_point() else v for k, v in net.state_dict().items()
    }
    other.load_state_dict(state)
    net.load_state_dict(state)
    with torch.no_grad():
        assert torch.equal(net(x), other(x))


def test_what_inference_keeps_follows_the_type():
    net = model("padded heads")
    x = image(16, 16)
    with torch.no_grad():
        full = net(x)
        half = net.half()(x.half())
        assert half.dtype == torch.float16
        again = net.float()(x)
    fresh = model("padded heads")
    with torch.no_grad():
        torch.testing.assert_close(half, fresh.half()(x.half()), atol=0, rtol=0)
    # FP32 weights rounded through FP16 stay close; nothing FP16 is reused in FP32
    assert again.dtype == full.dtype == torch.float32
    assert (again - full).abs().max() < 0.05


def test_a_copy_taken_in_inference_mode_runs_and_saves_plain_tensors():
    net = model("padded heads")
    x = image(16, 16)
    with torch.inference_mode():
        expected = net(x)
    copied = copy.deepcopy(net)
    with torch.no_grad():
        torch.testing.assert_close(copied(x), expected, atol=0, rtol=0)
    assert all(v.is_contiguous() for v in copied.state_dict().values())
    assert list(copied.state_dict()) == list(net.state_dict())


def test_a_trace_writes_the_traced_path():
    net = model("padded heads")
    x = image(16, 16)
    with torch.no_grad():
        traced = torch.jit.trace(net, x, check_trace=False)
        assert not getattr(net, "window_masks", None)
        assert not any(getattr(m, "derived", None) for m in net.modules())
        torch.testing.assert_close(traced(x), net(x), atol=2e-5, rtol=0)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"plus": True},
        {"upsampler": "pixelshuffle"},
        {"scale": 2, "shuffle_factor": 2},
    ],
    ids=["plain", "plus", "pixel shuffle", "pixel unshuffle"],
)
def test_esrgan_on_the_gpu_computes_the_same_in_either_layout(settings: dict):
    if not torch.cuda.is_available() or torch.version.cuda is None:
        pytest.skip("takes channels last only on CUDA")
    torch.manual_seed(0)
    net = ESRGAN(num_filters=16, num_blocks=2, **settings).eval().cuda()
    x = image(32, 48).cuda()
    with torch.no_grad():
        actual = net(x)
        expected = net.model(x) if not settings.get("shuffle_factor") else None
    assert actual.is_contiguous()
    if expected is not None:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
