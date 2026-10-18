import copy

import pytest
import torch
from torch import nn

from spandrel import MAIN_REGISTRY
from spandrel.architectures.HAT import HAT, HATArch
from spandrel.architectures.HAT.__arch.HAT import (
    OCAB,
    window_partition,
    window_reverse,
)

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


def _unfold_ocab(ocab: OCAB, x: torch.Tensor, h: int, w: int, rpi: torch.Tensor):
    """OCAB as it was before it cut its windows from blocks: nn.Unfold's windows."""
    b, _, c = x.shape
    ws, ows, heads = ocab.window_size, ocab.overlap_win_size, ocab.num_heads
    shortcut = x
    x = ocab.norm1(x).view(b, h, w, c)
    qkv = ocab.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)
    q = window_partition(qkv[0].permute(0, 2, 3, 1), ws).view(-1, ws * ws, c)
    unfold = nn.Unfold(kernel_size=(ows, ows), stride=ws, padding=(ows - ws) // 2)
    kv = unfold(torch.cat((qkv[1], qkv[2]), dim=1))
    kv = kv.view(b, 2, c, ows, ows, -1).permute(1, 0, 5, 3, 4, 2)
    kv = kv.reshape(2, -1, ows * ows, heads, c // heads).transpose(2, 3)
    q = q.reshape(-1, ws * ws, heads, c // heads).transpose(1, 2)
    attn = (q * ocab.scale) @ kv[0].transpose(-2, -1)
    attn = (attn + ocab.relative_position_bias(rpi)).softmax(dim=-1)
    out = (attn @ kv[1]).transpose(1, 2).reshape(-1, ws, ws, c)
    x = ocab.proj(window_reverse(out, ws, h, w).view(b, h * w, c)) + shortcut
    return x + ocab.mlp(ocab.norm2(x))


@pytest.mark.parametrize(
    ("window_size", "overlap_ratio"), [(16, 0.5), (13, 0.5), (16, 0.75), (8, 0.5)]
)
def test_overlapping_windows_match_unfold(window_size, overlap_ratio):
    torch.manual_seed(0)
    model = HAT(
        embed_dim=12,
        depths=(1,),
        num_heads=(2,),
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        compress_ratio=2,
        squeeze_factor=4,
    ).eval()
    ocab = model.get_submodule("layers.0.residual_group.overlap_attn")
    assert isinstance(ocab, OCAB)
    rpi = model.get_buffer("relative_position_index_OCA")
    h, w = 2 * window_size, 3 * window_size
    x = torch.rand(1, h * w, 12)
    with torch.no_grad():
        torch.testing.assert_close(
            ocab(x, (h, w), rpi), _unfold_ocab(ocab, x, h, w, rpi), rtol=0, atol=1e-6
        )
