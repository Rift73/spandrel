from collections.abc import Callable

import pytest
import torch
from torch import nn

from spandrel import ModelLoader
from spandrel.architectures.__arch_helpers.fp32_last_group import (
    FP32Conv2d,
    FP32LayerNorm,
)
from spandrel.architectures.DAT import DAT
from spandrel.architectures.HAT import HAT

MODELS: list[Callable[[], nn.Module]] = [
    lambda: DAT(embed_dim=30, depth=[2, 2, 2], num_heads=[2, 2, 2], split_size=[2, 4]),
    lambda: HAT(embed_dim=30, depths=(2, 2, 2), num_heads=(2, 2, 2), window_size=4),
]


def loaded(make: Callable[[], nn.Module]) -> nn.Module:
    torch.manual_seed(0)
    return ModelLoader().load_from_state_dict(make().state_dict()).model.eval()


def tile() -> torch.Tensor:
    return torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)


@pytest.mark.parametrize("make", MODELS, ids=["DAT", "HAT"])
def test_only_the_last_groups_are_wrapped(make: Callable[[], nn.Module]):
    net = loaded(make)
    assert isinstance(net.norm, FP32LayerNorm)
    layers = net.get_submodule("layers")
    assert isinstance(layers.get_submodule("0.conv"), nn.Conv2d)
    assert not isinstance(layers.get_submodule("0.conv"), FP32Conv2d)
    assert isinstance(layers.get_submodule("1.conv"), FP32Conv2d)
    assert isinstance(layers.get_submodule("2.conv"), FP32Conv2d)
    norms = [m for m in layers.get_submodule("2").modules() if isinstance(m, nn.LayerNorm)]
    wrapped = [m for m in norms if isinstance(m, FP32LayerNorm)]
    assert len(wrapped) >= 4
    assert not any(
        isinstance(m, FP32LayerNorm) for m in layers.get_submodule("0").modules()
    )


@pytest.mark.parametrize("make", MODELS, ids=["DAT", "HAT"])
def test_fp16_keeps_the_sums_that_outgrow_it_finite(make: Callable[[], nn.Module]):
    net = loaded(make)
    layers = net.get_submodule("layers")
    # As in trained models: each closing convolution fits FP16, their sum of about 90,000
    # in the second-to-last group does not.
    with torch.no_grad():
        for name, peak in (("0.conv", 30_000.0), ("1.conv", 60_000.0)):
            conv = layers.get_submodule(name)
            assert isinstance(conv, nn.Conv2d)
            conv.weight.zero_()
            assert conv.bias is not None
            conv.bias.copy_(torch.linspace(-peak, peak, conv.bias.numel()))
        half = net.half()
        dtypes: dict[str, torch.dtype] = {}
        for name in ("0", "1", "2"):
            layers.get_submodule(name).register_forward_hook(
                lambda _m, _i, o, name=name: dtypes.__setitem__(name, o.dtype)
            )
        output = half(tile().half())
    assert dtypes == {"0": torch.float16, "1": torch.float32, "2": torch.float32}
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("make", MODELS, ids=["DAT", "HAT"])
def test_the_state_dict_keeps_its_keys(make: Callable[[], nn.Module]):
    assert list(loaded(make).state_dict()) == list(make().state_dict())
