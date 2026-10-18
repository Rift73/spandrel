import copy

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.HAT import HAT, HATArch

from .util import (
    ModelFile,
    TestImage,
    assert_image_inference,
    assert_loads_correctly,
    assert_size_requirements,
    assert_traced_padding,
    disallowed_props,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        HATArch(),
        lambda: HAT(),
        lambda: HAT(in_chans=1),
        lambda: HAT(in_chans=4),
        lambda: HAT(embed_dim=64),
        lambda: HAT(depths=(1, 2, 3, 2, 1), num_heads=(5, 4, 3, 5, 6)),
        lambda: HAT(window_size=13),
        lambda: HAT(compress_ratio=2),
        lambda: HAT(squeeze_factor=15),
        lambda: HAT(overlap_ratio=0.75),
        lambda: HAT(mlp_ratio=3),
        lambda: HAT(qkv_bias=False),
        lambda: HAT(ape=True),
        lambda: HAT(patch_norm=False),
        lambda: HAT(resi_connection="1conv"),
        lambda: HAT(resi_connection="identity"),
        lambda: HAT(upsampler="pixelshuffle", upscale=1),
        lambda: HAT(upsampler="pixelshuffle", upscale=2),
        lambda: HAT(upsampler="pixelshuffle", upscale=3),
        lambda: HAT(upsampler="pixelshuffle", upscale=4),
        lambda: HAT(upsampler="pixelshuffle", upscale=8),
        lambda: HAT(upsampler="pixelshuffle", num_feat=32),
        # there are multiple equivalent squeeze_factor values for each models,
        # so we cannot test for equal value
        ignore_parameters={"squeeze_factor"},
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/hat/2x-HAT-S_SR.pth",
        name="HAT-S_SRx2.pth",
    )
    assert_size_requirements(file.load_model())

    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/hat/4x-HAT_SR.pth",
        name="HAT_SRx4.pth",
    )
    assert_size_requirements(file.load_model())


def test_traced_padding():
    # the exported graph only ran at multiples of the window size
    assert_traced_padding(
        HATArch().load(
            HAT(
                embed_dim=48,
                depths=(2,),
                num_heads=(2,),
                window_size=16,
                squeeze_factor=8,
                upscale=2,
                upsampler="pixelshuffle",
            ).state_dict()
        )
    )


def test_HAT_S_2x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/hat/2x-HAT-S_SR.pth",
        name="HAT-S_SRx2.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_S_3x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/hat/3x-HAT-S_SR.pth",
        name="HAT-S_SRx3.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_S_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/hat/4x-HAT-S_SR.pth",
        name="HAT-S_SRx4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_3x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/hat/3x-HAT_SR.pth",
        name="HAT_SRx3.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/hat/4x-HAT_SR.pth",
        name="HAT_SRx4.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_L_4x(snapshot):
    file = ModelFile.from_url(
        "https://github.com/OpenModelDB/model-hub/releases/download/hat/4x-HAT-L_SR_ImageNet-pretrain.pth",
        name="HAT-L_SRx4_ImageNet-pretrain.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_community1(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/4xLexicaHAT/4xLexicaHAT.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


def test_HAT_community2(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/raw/main/4xNomos8kSCHAT-S/4xNomos8kSCHAT-S.pth"
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, HAT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32, TestImage.SR_64],
    )


@pytest.mark.parametrize("upscale", [1, 2])
def test_half_survives_a_last_group_beyond_fp16(upscale):
    torch.manual_seed(0)
    model = HAT(
        embed_dim=24,
        depths=(2, 2),
        num_heads=(2, 2),
        window_size=8,
        compress_ratio=2,
        squeeze_factor=4,
        upsampler="pixelshuffle",
        upscale=upscale,
    ).eval()
    # A trained 4x HAT-L writes about 722k in its last group's closing conv,
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
