import pytest
import torch

from spandrel import MAIN_REGISTRY, ImageModelDescriptor
from spandrel.architectures.MoSR import MoSR
from spandrel.architectures.SMoSR import SMoSR, SMoSRArch

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)

# SMoSR has no presets; the source repo holds no configs. With rep, the conv in
# front of DySample outputs out_ch instead of upsampler_mid_dim, so rep and
# dysample only run together when upsampler_mid_dim is DySample's input width
# (dim + in_ch * scale**2), which drops that conv.
VARIANTS = [
    {},
    {"rep": True},
    {"scale": 1},
    {"scale": 1, "rep": True},
    # at scale 1 only the MetaUpsample buffer keeps the upsampler
    {"scale": 1, "upsampler": "pa_up"},
    {"scale": 3},
    {"scale": 4},
    {"scale": 4, "rep": True},
    *(
        {"upsampler": upsampler, "scale": scale, "rep": rep}
        for upsampler in ("pixelshuffle", "nearest+conv", "pa_up")
        for scale in (2, 3, 4)
        for rep in (False, True)
    ),
    *({"upsampler": "dysample", "scale": scale} for scale in (2, 3, 4)),
    {"upsampler": "dysample", "upsampler_mid_dim": 60},
    {"upsampler": "dysample", "upsampler_mid_dim": 60, "rep": True},
    {"upsampler": "dysample", "upsampler_mid_dim": 16},
    {"upsampler": "dysample", "d_kernel": 1},
    {"upsampler": "dysample", "d_kernel": 5},
    {"upsampler": "pixelshuffle", "upsampler_mid_dim": 64},
    {"upsampler": "pa_up", "upsampler_mid_dim": 16},
    # kept only in the upsampler's MetaUpsample buffer
    {"upsampler_mid_dim": 48},
    {"upsampler": "nearest+conv", "upsampler_mid_dim": 64},
    {"in_ch": 1, "out_ch": 1},
    {"in_ch": 4, "out_ch": 4},
    {"in_ch": 1, "out_ch": 3},
    {"out_ch": 1, "upsampler": "pixelshuffle"},
    {"out_ch": 1, "upsampler": "dysample"},
    {"in_ch": 1, "out_ch": 1, "upsampler": "pa_up", "scale": 4},
    {"dim": 64},
    {"dim": 32, "rep": True},
    {"n_mb": 0},
    {"n_mb": 6},
]

# MoSR shares its name and DySample code with SMoSR (cf. test_MoSR.py)
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
    model = SMoSR(**kwargs)
    # taken in training mode, as a trainer saves it: every eval_conv still
    # holds its initial weights, not the folded DOConv2d/ConvNXC ones
    state_dict = model.state_dict()
    in_ch = model.hyperparameters["in_ch"]
    out_ch = model.hyperparameters["out_ch"]
    scale = model.hyperparameters["scale"]

    # no other architecture, MoSR and GaterV3 included, claims an SMoSR
    assert _claimants(state_dict) == ["SMoSR"]
    descriptor = MAIN_REGISTRY.load(state_dict)
    assert isinstance(descriptor, ImageModelDescriptor)
    assert descriptor.architecture.id == "SMoSR"
    assert isinstance(descriptor.model, SMoSR)
    assert descriptor.scale == scale
    assert descriptor.input_channels == in_ch
    assert descriptor.output_channels == out_ch
    assert descriptor.model.hyperparameters == model.hyperparameters

    loaded = descriptor.model.state_dict()
    assert list(loaded) == list(state_dict)
    for key, value in state_dict.items():
        assert loaded[key].shape == value.shape, key

    image = torch.rand(1, in_ch, 13, 17)
    with torch.no_grad():
        expected = model.eval()(image)
        actual = descriptor.model.eval()(image)
    assert actual.shape == (1, out_ch, 13 * scale, 17 * scale)
    torch.testing.assert_close(actual, expected)

    del descriptor, loaded
    # now in eval mode: the folded state dict, also through safetensors
    assert_loads_correctly(SMoSRArch(), lambda: model)


@pytest.mark.parametrize("kwargs", MOSR_VARIANTS, ids=str)
def test_mosr_is_not_claimed(kwargs):
    state_dict = MoSR(**kwargs).state_dict()
    assert not SMoSRArch().detect(state_dict)
    assert MAIN_REGISTRY.load(state_dict).architecture.id == "MoSR"


def test_size_requirements():
    torch.manual_seed(0)
    descriptor = MAIN_REGISTRY.load(SMoSR(rep=True).state_dict())
    assert isinstance(descriptor, ImageModelDescriptor)
    assert descriptor.size_requirements.minimum == 3
    assert_size_requirements(descriptor)
    # inputs too small for the model's own reflect padding
    assert descriptor(torch.rand(1, 3, 1, 2)).shape == (1, 3, 2, 4)
