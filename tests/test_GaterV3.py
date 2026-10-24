import pytest
import torch

from spandrel import MAIN_REGISTRY, ImageModelDescriptor
from spandrel.architectures.GaterV3 import GateRV3, GaterV3Arch
from spandrel.architectures.MoSR import MoSR

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)

# The source repo holds no presets or configs; traiNNer-redux registers gaterv3_r (the
# defaults) and gaterv3_s (three levels).
VARIANTS = [
    {},
    {"enc_blocks": (2, 2, 4), "dec_blocks": (2, 2, 2)},
    {"enc_blocks": (2, 2, 4), "dec_blocks": (2, 2, 2), "scale": 4},
    {"scale": 1},
    {"scale": 3},
    {"scale": 4},
    *(
        {"upsample": upsample, "scale": scale}
        for upsample in ("pixelshuffledirect", "nearest+conv", "dysample")
        for scale in (2, 3, 4)
    ),
    {"upsample_mid_dim": 32},
    {"scale": 3, "upsample_mid_dim": 64},
    # kept only in the upsampler's MetaUpsample buffer
    {"upsample": "dysample", "upsample_mid_dim": 64},
    {"upsample": "nearest+conv", "scale": 3, "upsample_mid_dim": 16},
    {"in_ch": 1},
    {"in_ch": 4, "upsample": "dysample", "scale": 4},
    # wider models with fewer blocks: the default depth at dim 64 has 89M weights
    {"dim": 48, "enc_blocks": (1, 1, 1, 1), "num_latent": 2},
    {"dim": 64, "scale": 4, "enc_blocks": (1, 1, 1, 1), "num_latent": 1},
    {"enc_blocks": (1, 2, 3), "dec_blocks": (1, 1, 2)},
    {"enc_blocks": (2, 2), "dec_blocks": (1, 3)},
    {"enc_blocks": (0, 1, 0, 2), "dec_blocks": (1, 0, 1, 0)},
    {"num_latent": 2},
    {"num_latent": 0},
    {"attention": True},
    {"attention": True, "flash": False},
    {"sisr_blocks": 0},
    {"sisr_blocks": 2},
]

# MoSR shares GatedCNNBlock and DySample code with GaterV3 (cf. test_MoSR.py)
MOSR_VARIANTS = [
    {},
    {"in_ch": 1, "out_ch": 1},
    {"n_block": 5},
    {"dim": 48},
    {"upscale": 2},
    {"upsampler": "dys"},
    {"upsampler": "gps"},
    {"expansion_ratio": 2.0},
    {"conv_ratio": 2.0},
]


def _claimants(state_dict):
    return [
        support.architecture.id
        for support in MAIN_REGISTRY.architectures(order="detection")
        if support.detect(state_dict)
    ]


@pytest.mark.parametrize("kwargs", VARIANTS, ids=str)
def test_variant(kwargs):
    torch.manual_seed(0)
    model = GateRV3(**kwargs)
    # taken in training mode, as a trainer saves it: eval_conv still holds its
    # initial weights, not the fused SPAB branches
    state_dict = model.state_dict()
    in_ch = model.hyperparameters["in_ch"]
    scale = model.hyperparameters["scale"]

    # no other architecture, MoSR and SMoSR included, claims a GaterV3
    assert _claimants(state_dict) == ["GaterV3"]
    descriptor = MAIN_REGISTRY.load(state_dict)
    assert isinstance(descriptor, ImageModelDescriptor)
    assert descriptor.architecture.id == "GaterV3"
    assert isinstance(descriptor.model, GateRV3)
    assert descriptor.scale == scale
    assert descriptor.input_channels == in_ch
    assert descriptor.output_channels == in_ch
    assert descriptor.model.hyperparameters == model.hyperparameters

    loaded = descriptor.model.state_dict()
    assert list(loaded) == list(state_dict)
    for key, value in state_dict.items():
        assert loaded[key].shape == value.shape, key

    image = torch.rand(1, in_ch, 20, 24)
    with torch.no_grad():
        expected = model.eval()(image)
        actual = descriptor.model.eval()(image)
    assert actual.shape == (1, in_ch, 20 * scale, 24 * scale)
    torch.testing.assert_close(actual, expected)

    del descriptor, loaded
    # now in eval mode: the fused state dict, also through safetensors
    assert_loads_correctly(GaterV3Arch(), lambda: model)


@pytest.mark.parametrize("kwargs", MOSR_VARIANTS, ids=str)
def test_mosr_is_not_claimed(kwargs):
    state_dict = MoSR(**kwargs).state_dict()
    assert not GaterV3Arch().detect(state_dict)
    assert MAIN_REGISTRY.load(state_dict).architecture.id == "MoSR"


def test_size_requirements():
    torch.manual_seed(0)
    model = GateRV3(enc_blocks=(1, 1, 1, 1), dec_blocks=(1, 1, 1, 1))
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert isinstance(descriptor, ImageModelDescriptor)
    assert descriptor.size_requirements.multiple_of == 16
    assert_size_requirements(descriptor)

    # spandrel pads to the multiple the way the model pads itself
    image = torch.rand(1, 3, 19, 23)
    with torch.no_grad():
        expected = model.eval()(image).clamp(0, 1)
    torch.testing.assert_close(descriptor(image), expected)
    # inputs too small for the model's own reflect padding
    assert descriptor(torch.rand(1, 3, 5, 7)).shape == (1, 3, 10, 14)


def test_three_levels_need_a_multiple_of_eight():
    model = GateRV3(enc_blocks=(1, 1, 1), dec_blocks=(1, 1, 1))
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert descriptor.size_requirements.multiple_of == 8
