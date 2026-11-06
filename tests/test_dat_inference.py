from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch
from torch import nn

from spandrel import ModelLoader
from spandrel.architectures.DAT import DAT
from spandrel.architectures.DAT.__arch import DAT as DAT_MODULE

MODELS: dict[str, Callable[[], nn.Module]] = {
    "plain": lambda: DAT(
        embed_dim=30, depth=[3, 3], num_heads=[2, 2], split_size=[2, 4]
    ),
    "wide windows": lambda: DAT(
        embed_dim=48, depth=[3, 3], num_heads=[4, 4], split_size=[4, 8], qkv_bias=True
    ),
    "3conv": lambda: DAT(
        embed_dim=30,
        depth=[3, 3],
        num_heads=[2, 2],
        split_size=[2, 4],
        resi_connection="3conv",
    ),
    "direct": lambda: DAT(
        embed_dim=30,
        depth=[3, 3],
        num_heads=[2, 2],
        split_size=[2, 4],
        upsampler="pixelshuffledirect",
    ),
}
# Multiples of the windows and not, wider than tall and taller than wide, one window's size
SIZES = [(16, 16), (24, 40), (40, 24), (21, 37), (8, 8), (9, 33)]


def model(name: str) -> nn.Module:
    torch.manual_seed(0)
    return ModelLoader().load_from_state_dict(MODELS[name]().state_dict()).model.eval()


def image(height: int, width: int, batch: int = 1) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(batch, 3, height, width)


def traced_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The channels-first path an ONNX export traces, with nothing kept."""
    monkeypatch.setattr(DAT_MODULE, "_traced", lambda: True)


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


def test_gradients_take_the_same_values_and_keep_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    net = model("plain")
    x = image(16, 16)
    actual = net(x)
    with torch.no_grad(), monkeypatch.context() as patch:
        traced_path(patch)
        expected = net(x)
    torch.testing.assert_close(actual.detach(), expected, atol=2e-5, rtol=0)
    actual.sum().backward()
    assert not any(getattr(m, "derived", None) for m in net.modules())
    assert not any(getattr(m, "border_masks", None) for m in net.modules())


def test_a_run_keeps_one_set_of_masks_for_all_blocks():
    net = model("plain")
    with torch.no_grad():
        net(image(16, 16))
    stores = {id(m.border_masks) for m in net.modules() if hasattr(m, "border_masks")}
    assert len(stores) == 1
    assert any(m.border_masks for m in net.modules() if hasattr(m, "border_masks"))


def test_what_inference_keeps_follows_new_weights():
    net = model("plain")
    x = image(16, 16)
    with torch.no_grad():
        net(x)
    other = model("plain")
    state = {
        k: v + 0.01 if v.is_floating_point() else v for k, v in net.state_dict().items()
    }
    other.load_state_dict(state)
    net.load_state_dict(state)
    with torch.no_grad():
        assert torch.equal(net(x), other(x))


def test_what_inference_keeps_follows_the_type():
    net = model("plain")
    x = image(16, 16)
    with torch.no_grad():
        full = net(x)
        half = net.half()(x.half())
        assert half.dtype == torch.float16
        again = net.float()(x)
    fresh = model("plain")
    with torch.no_grad():
        torch.testing.assert_close(half, fresh.half()(x.half()), atol=0, rtol=0)
    # FP32 weights rounded through FP16 stay close; nothing FP16 is reused in FP32
    assert again.dtype == full.dtype == torch.float32
    assert (again - full).abs().max() < 0.05


def test_a_copy_taken_in_inference_mode_runs_and_saves_plain_tensors():
    net = model("plain")
    x = image(16, 16)
    with torch.inference_mode():
        expected = net(x)
    copied = copy.deepcopy(net)
    with torch.no_grad():
        torch.testing.assert_close(copied(x), expected, atol=0, rtol=0)
    assert all(v.is_contiguous() for v in copied.state_dict().values())
    assert list(copied.state_dict()) == list(net.state_dict())
