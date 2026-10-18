import copy

import pytest
import torch

from spandrel import MAIN_REGISTRY, ModelLoader
from spandrel.architectures.DAT import DAT, DATArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        DATArch(),
        lambda: DAT(),
        lambda: DAT(embed_dim=60),
        lambda: DAT(in_chans=1),
        lambda: DAT(in_chans=4),
        lambda: DAT(depth=[2, 3], num_heads=[2, 5]),
        lambda: DAT(depth=[2, 3, 4, 2], num_heads=[2, 3, 2, 2]),
        lambda: DAT(depth=[2, 3, 4, 2, 5], num_heads=[2, 3, 2, 2, 3]),
        lambda: DAT(upsampler="pixelshuffle", upscale=1),
        lambda: DAT(upsampler="pixelshuffle", upscale=2),
        lambda: DAT(upsampler="pixelshuffle", upscale=3),
        lambda: DAT(upsampler="pixelshuffle", upscale=4),
        lambda: DAT(upsampler="pixelshuffle", upscale=8),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=1),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=2),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=3),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=4),
        lambda: DAT(upsampler="pixelshuffledirect", upscale=8),
        lambda: DAT(resi_connection="3conv"),
        lambda: DAT(qkv_bias=False),
        lambda: DAT(split_size=[4, 4]),
        lambda: DAT(split_size=[2, 8]),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/dat/4x-DAT_S.pth"
    )
    assert_size_requirements(file.load_model())


def test_DAT_S_x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/dat/4x-DAT_S.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DAT_S_x3(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/dat/3x-DAT_S.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DAT_x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/dat/4x-DAT.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DAT_2_x4(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/dat/4x-DAT_2.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_DAT_tags(snapshot):
    loader = ModelLoader()

    # https://github.com/muslll/neosr/blob/master/neosr/archs/dat_arch.py#L876

    dat_light = DAT(
        in_chans=3,
        img_range=1.0,
        depth=[18],
        embed_dim=60,
        num_heads=[6],
        expansion_factor=2,
        resi_connection="3conv",
        split_size=[8, 32],
        upsampler="pixelshuffledirect",
    )
    assert loader.load_from_state_dict(dat_light.state_dict()) == snapshot(
        exclude=disallowed_props
    )

    dat_small = DAT(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 16],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=2,
        resi_connection="1conv",
    )
    assert loader.load_from_state_dict(dat_small.state_dict()) == snapshot(
        exclude=disallowed_props
    )

    dat_medium = DAT(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 32],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=4,
        resi_connection="1conv",
    )
    assert loader.load_from_state_dict(dat_medium.state_dict()) == snapshot(
        exclude=disallowed_props
    )

    dat_2 = DAT(
        in_chans=3,
        img_range=1.0,
        split_size=[8, 32],
        depth=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        expansion_factor=2,
        resi_connection="1conv",
    )
    assert loader.load_from_state_dict(dat_2.state_dict()) == snapshot(
        exclude=disallowed_props
    )


@pytest.mark.parametrize("resi_connection", ["1conv", "3conv"])
def test_half_survives_a_last_group_beyond_fp16(resi_connection):
    torch.manual_seed(0)
    model = DAT(
        embed_dim=24,
        depth=[2, 2],
        num_heads=[2, 2],
        upsampler="pixelshuffle",
        upscale=2,
        resi_connection=resi_connection,
    ).eval()
    # A trained 4x DAT2 writes about 121k in its last group's closing conv,
    # beyond FP16's 65504; the final norm right after it scales that back down.
    for parameter in model.layers[-1].get_submodule("conv").parameters():
        parameter.data.mul_(1e5)
    image = torch.rand(1, 3, 24, 20)
    with torch.no_grad():
        expected = model(image)
        assert not torch.isfinite(copy.deepcopy(model).half()(image.half())).all()

    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert descriptor.supports_half
    assert descriptor.model.state_dict().keys() == model.state_dict().keys()
    with torch.no_grad():
        actual = descriptor.model.eval().half()(image.half()).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-3)
