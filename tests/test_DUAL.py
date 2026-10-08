import copy

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.DUAL import DUAL, DUALArch
from spandrel.architectures.DUAL.__arch.dual_arch import dual_light, dual_xs

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)


def _small(**kwargs):
    return DUAL(
        embed_dim=32,
        depths=(4, 4),
        layer_indices=((0, 1, 2, 5), (2, 3, 4, 5)),
        **kwargs,
    )


def test_load():
    assert_loads_correctly(
        DUALArch(),
        lambda: dual_light(4),
        lambda: dual_xs(4),
        lambda: dual_light(2),
        lambda: dual_xs(1),
        lambda: DUAL(embed_dim=32, depths=(6, 6)),
        lambda: DUAL(scale=2, embed_dim=64, depths=(6, 6, 6)),
        lambda: _small(scale=1),
        lambda: _small(scale=2, pixel_unshuffle=True),
        lambda: _small(c_pos_bias="rib"),
        lambda: _small(c_pos_bias="rib", rfb_rank=8),
        lambda: _small(c_pos_bias="none"),
        lambda: _small(rfb_rank=8),
        lambda: _small(qk="cosine"),
        lambda: _small(g_window=8),
        lambda: _small(dynamic_experts=2),
        lambda: DUAL(embed_dim=32, depths=(2, 4), layer_indices=((0, 1), (2, 3, 4, 5))),
    )


def test_ambiguous_schedule_is_refused():
    # (3, 4, 5) and (0, 1, 2) both put the context layer last
    model = DUAL(embed_dim=32, depths=(3, 3), layer_indices=((0, 1, 2), (3, 4, 5)))
    with pytest.raises(ValueError, match="layer schedule"):
        DUALArch().load(model.state_dict())


@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize("pixel_unshuffle", [False, True])
def test_loaded_model_matches(folded, pixel_unshuffle):
    torch.manual_seed(0)
    model = _small(scale=2, pixel_unshuffle=pixel_unshuffle).eval()
    # the expert kernels start at zero; give the routed path some weight
    for name, parameter in model.named_parameters():
        if name.endswith("conv.experts"):
            parameter.data.normal_(0, 0.02)
    source = copy.deepcopy(model)
    if folded:
        source.prepare_for_export()
    descriptor = MAIN_REGISTRY.load(source.state_dict())
    assert descriptor.architecture.id == "DUAL"
    assert descriptor.scale == 2

    image = torch.rand(1, 3, 37, 29)
    with torch.no_grad():
        expected = model(image)
        actual = descriptor.model(image)
    assert actual.shape == (1, 3, 74, 58)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_size_requirements():
    assert_size_requirements(DUALArch().load(_small(scale=2).state_dict()))
    assert_size_requirements(DUALArch().load(_small(pixel_unshuffle=True).state_dict()))
