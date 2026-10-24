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

# resselt's gaterv3.3 writes MetaUpsample version 3: three more upsamplers, a conv
# in front of DySample and LDA when upsample_mid_dim differs from dim, and
# DySample's end_kernel. A small trunk keeps these quick.
_V3 = {
    "upsample_version": 3,
    "enc_blocks": (1, 1, 1, 1),
    "dec_blocks": (1, 1, 1, 1),
    "num_latent": 1,
    "sisr_blocks": 1,
}
RESSELT_VARIANTS = [
    *(
        {**_V3, "upsample": upsample, "scale": scale}
        for upsample in ("transpose+conv", "lda", "pa_up")
        for scale in (2, 3, 4)
    ),
    *(
        {
            **_V3,
            "upsample": "dysample",
            "scale": scale,
            "upsample_mid_dim": mid,
            "end_kernel": end_kernel,
        }
        for scale, mid, end_kernel in ((2, 32, 3), (3, 48, 1), (4, 64, 3))
    ),
    {**_V3, "upsample": "lda", "upsample_mid_dim": 32},
    {**_V3, "upsample": "pa_up", "scale": 4, "upsample_mid_dim": 64},
    {**_V3, "upsample": "pa_up", "scale": 3, "upsample_mid_dim": 16},
    {**_V3, "upsample": "pixelshuffle", "scale": 3, "upsample_mid_dim": 32},
    {**_V3, "upsample": "pixelshuffledirect", "scale": 4},
    {**_V3, "upsample": "nearest+conv", "scale": 3},
    {**_V3, "in_ch": 1, "upsample": "lda"},
    {**_V3, "in_ch": 4, "upsample": "transpose+conv", "scale": 4},
    {**_V3, "attention": True, "flash": False, "upsample": "pa_up"},
    # the layout of the released 1x models
    {"enc_blocks": (2, 2, 4, 8), "num_latent": 12, "scale": 1, "attention": True},
]
_META = "dim_to_in.MetaUpsample"

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


@pytest.mark.parametrize("kwargs", VARIANTS + RESSELT_VARIANTS, ids=str)
def test_variant(kwargs):
    torch.manual_seed(0)
    model = GateRV3(**kwargs)
    # taken in training mode, as a trainer saves it: eval_conv still holds its
    # initial weights, not the fused SPAB branches
    state_dict = model.state_dict()
    in_ch = model.hyperparameters["in_ch"]
    scale = model.hyperparameters["scale"]
    # loading replaces the buffer in the state dict with the model's own
    meta = state_dict[_META].clone() if scale != 1 else None

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
    if meta is not None:
        # the same upsampler record, version included
        assert torch.equal(loaded[_META], meta)

    image = torch.rand(1, in_ch, 20, 24)
    with torch.no_grad():
        expected = model.eval()(image)
        actual = descriptor.model.eval()(image)
    assert actual.shape == (1, in_ch, 20 * scale, 24 * scale)
    torch.testing.assert_close(actual, expected)

    del descriptor, loaded
    # now in eval mode: the fused state dict, also through safetensors
    assert_loads_correctly(GaterV3Arch(), lambda: model)


@pytest.mark.parametrize(
    ("with_gamma", "kwargs"),
    [
        (False, {}),
        (True, {}),
        (True, {"scale": 2, "upsample": "lda", "upsample_version": 3}),
    ],
    ids=str,
)
def test_checkpoint_without_flash_buffers(with_gamma, kwargs):
    # The first gaterv3.py and resselt's gaterv3.3 have no flash buffer and always
    # run channel attention; the first also has no gamma. Released checkpoints
    # have neither, or gamma alone.
    torch.manual_seed(0)
    model = GateRV3(attention=True, flash=False, num_latent=2, **kwargs).eval()
    state_dict = {
        key: value
        for key, value in model.state_dict().items()
        if not key.endswith("token_mix.flash") and (with_gamma or key != "gamma")
    }

    descriptor = MAIN_REGISTRY.load(state_dict)
    assert isinstance(descriptor, ImageModelDescriptor)
    assert isinstance(descriptor.model, GateRV3)
    assert descriptor.model.hyperparameters["flash"] is False
    image = torch.rand(1, 3, 20, 24)
    with torch.no_grad():
        torch.testing.assert_close(descriptor.model.eval()(image), model(image))


def test_dysample_versions():
    # gaterv3.2 (version 1) only records upsample_mid_dim; resselt's gaterv3.3
    # (version 3) puts a conv in front of DySample when it differs from dim
    for version, end_kernel in ((1, 1), (3, 3)):
        state_dict = GateRV3(
            upsample="dysample",
            upsample_mid_dim=48,
            end_kernel=end_kernel,
            upsample_version=version,
            enc_blocks=(1, 1),
            dec_blocks=(1, 1),
            num_latent=1,
        ).state_dict()
        if version == 1:
            assert "dim_to_in.0.weight" not in state_dict
            assert state_dict["dim_to_in.0.end_conv.weight"].shape[2] == 1
        else:
            assert state_dict["dim_to_in.0.weight"].shape[0] == 48
            assert state_dict["dim_to_in.2.end_conv.weight"].shape[2] == 3

        model = MAIN_REGISTRY.load(state_dict).model
        assert isinstance(model, GateRV3)
        assert model.hyperparameters["upsample_version"] == version
        assert model.hyperparameters["upsample_mid_dim"] == 48
        assert model.hyperparameters["end_kernel"] == end_kernel


def test_conv_upsampler_cannot_upscale():
    # one conv at the input size cannot be added to the upscaled input, in
    # resselt's code too, so no such model runs
    model = GateRV3(
        scale=2, upsample="conv", enc_blocks=(1, 1), dec_blocks=(1, 1), num_latent=1
    )
    with pytest.raises(ValueError, match="conv upsampler"):
        MAIN_REGISTRY.load(model.state_dict())


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
