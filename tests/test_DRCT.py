from spandrel.architectures.DRCT import DRCT, DRCTArch

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
        DRCTArch(),
        lambda: DRCT(),
        lambda: DRCT(in_chans=4, embed_dim=60),
        lambda: DRCT(upsampler="pixelshuffle", upscale=2, resi_connection="1conv"),
        lambda: DRCT(upsampler="pixelshuffle", upscale=4, resi_connection="1conv"),
        lambda: DRCT(upsampler="", upscale=1, resi_connection="identity"),
        lambda: DRCT(qkv_bias=False),
        lambda: DRCT(gc=16),
        lambda: DRCT(ape=True, patch_norm=False),
        lambda: DRCT(mlp_ratio=4.0),
        lambda: DRCT(window_size=8),
        lambda: DRCT(depths=[6, 6, 6, 6], num_heads=[6, 4, 6, 3]),
        lambda: DRCT(img_size=32),
        lambda: DRCT(img_size=16),
    )


def test_size_requirements():
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_drct-l/4xRealWebPhoto_v4_drct-l.pth"
    )
    assert_size_requirements(file.load_model())


def test_traced_padding():
    # the exported graph skipped the padding to multiples of 16
    assert_traced_padding(
        DRCTArch().load(
            DRCT(
                embed_dim=30,
                depths=(2,),
                num_heads=(2,),
                window_size=8,
                upscale=2,
                upsampler="pixelshuffle",
            ).state_dict()
        )
    )


def test_community_model(snapshot):
    file = ModelFile.from_url(
        "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_drct-l/4xRealWebPhoto_v4_drct-l.pth",
    )
    model = file.load_model()
    assert model == snapshot(exclude=disallowed_props)
    assert isinstance(model.model, DRCT)
    assert_image_inference(
        file,
        model,
        [TestImage.SR_16, TestImage.SR_32],
    )
