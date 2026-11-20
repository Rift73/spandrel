from __future__ import annotations

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.FDAT import FDAT
from spandrel.architectures.FDAT.__arch.fdat import SimplifiedDATBlock
from spandrel.architectures.TFDAT import TFDAT, TFDATArch, TFDATv2, TFDATv2Arch
from spandrel.architectures.TFDAT.__arch.tfdat import warp_frame

from .util import assert_loads_correctly, skip_if_unchanged

skip_if_unchanged(__file__)

Version = type[TFDAT] | type[TFDATv2]


def _small(version: Version, scale: int = 2, clip_size: int = 5) -> TFDAT | TFDATv2:
    """2 groups of 4 blocks: a group has a shifted spatial block."""
    return version(
        scale=scale,
        clip_size=clip_size,
        embed_dim=32,
        num_groups=2,
        depth_per_group=2,
        num_heads=2,
        mid_dim=32,
        flow_base_ch=16,
    )


def _random(model: torch.nn.Module) -> torch.nn.Module:
    """Every parameter random (the trainer zero-initializes the flow heads and other
    layers, which makes the flow the identity)."""
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            std = parameter[0].numel() ** -0.5 if parameter.dim() > 1 else 0.2
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)
    return model.eval()


def _clip(clip_size: int, height: int = 27, width: int = 45) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.rand(1, clip_size, 3, height, width, generator=generator)


def test_load():
    assert_loads_correctly(
        TFDATArch(),
        lambda: TFDAT(),
        lambda: TFDAT(scale=1),
        lambda: TFDAT(scale=3),
        lambda: TFDAT(scale=4, upsampler_type="transpose+conv"),
        lambda: TFDAT(scale=4, upsampler_type="dysample", mid_dim=48),
        lambda: TFDAT(scale=2, upsampler_type="pa_up"),
        lambda: TFDAT(scale=2, upsampler_type="nearest+conv"),
        lambda: TFDAT(scale=2, upsampler_type="pixelshuffledirect"),
        lambda: TFDAT(num_in_ch=1, num_out_ch=1),
        lambda: TFDAT(embed_dim=120, num_groups=6, depth_per_group=2),
        lambda: TFDAT(num_heads=6, embed_dim=96, window_size=16),
        lambda: TFDAT(ffn_expansion_ratio=1.5, aim_reduction_ratio=4),
        lambda: TFDAT(group_block_pattern=["spatial", "spatial", "channel"]),
        lambda: TFDAT(flow_base_ch=24),
    )
    assert_loads_correctly(
        TFDATv2Arch(),
        lambda: TFDATv2(),
        lambda: TFDATv2(scale=1),
        lambda: TFDATv2(scale=3),
        lambda: TFDATv2(scale=4, upsampler_type="transpose+conv"),
        lambda: TFDATv2(scale=4, upsampler_type="dysample", mid_dim=48),
        lambda: TFDATv2(scale=2, upsampler_type="pa_up"),
        lambda: TFDATv2(clip_size=3),
        lambda: TFDATv2(clip_size=9, embed_dim=120),
        lambda: TFDATv2(num_in_ch=1, num_out_ch=1),
        lambda: TFDATv2(num_in_ch=3, num_out_ch=1),
        lambda: TFDATv2(num_groups=6, depth_per_group=2),
        lambda: TFDATv2(num_heads=6, embed_dim=96, window_size=16),
        lambda: TFDATv2(ffn_expansion_ratio=1.5, aim_reduction_ratio=4),
        lambda: TFDATv2(group_block_pattern=["spatial", "spatial", "channel"]),
        lambda: TFDATv2(flow_base_ch=24),
    )


def test_registry_tells_the_versions_and_fdat_apart():
    v1 = _small(TFDAT).state_dict()
    v2 = _small(TFDATv2).state_dict()
    for scale in (2, 4):
        fdat = FDAT(scale=scale, embed_dim=32, num_groups=2).state_dict()
        assert MAIN_REGISTRY.load(fdat).architecture.id == "FDAT"
        assert not TFDATArch().detect(fdat)
        assert not TFDATv2Arch().detect(fdat)
    fdat = FDAT(scale=2, unshuffle_mod=True).state_dict()
    assert MAIN_REGISTRY.load(fdat).architecture.id == "FDAT"
    assert not TFDATv2Arch().detect(fdat)
    assert MAIN_REGISTRY.load(v1).architecture.id == "TFDAT"
    assert MAIN_REGISTRY.load(v2).architecture.id == "TFDATv2"
    assert not TFDATv2Arch().detect(v1)
    assert not TFDATArch().detect(v2)


@pytest.mark.parametrize("version", [TFDAT, TFDATv2])
def test_other_architectures_do_not_claim_tfdat(version: Version):
    state_dict = _small(version).state_dict()
    own = "TFDAT" if version is TFDAT else "TFDATv2"
    claimed = [
        support.architecture.id
        for support in MAIN_REGISTRY.architectures()
        # FDAT comes after TFDAT: a TFDAT file holds every key FDAT checks for.
        if support.architecture.id not in (own, "FDAT") and support.detect(state_dict)
    ]
    assert claimed == []


def test_shifted_windows():
    # every second spatial block of a group: blocks 0 and 2 of spatial, channel, ...
    def shifts(model: TFDAT | TFDATv2) -> list[int]:
        blocks = [m for m in model.modules() if isinstance(m, SimplifiedDATBlock)]
        return [getattr(block.attn, "shift", -1) for block in blocks[:4]]

    assert shifts(_small(TFDATv2)) == [0, -1, 4, -1]
    assert shifts(_small(TFDAT)) == [0, -1, 0, -1]


@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("size", [(27, 45), (32, 64)], ids=["padded", "aligned"])
def test_output_is_the_centre_frame_upscaled(scale: int, size: tuple[int, int]):
    for version in (TFDAT, TFDATv2):
        descriptor = MAIN_REGISTRY.load(_random(_small(version, scale)).state_dict())
        frames = _clip(5, *size)
        with torch.inference_mode():
            out = descriptor.model(frames)
        assert out.shape == (1, 3, size[0] * scale, size[1] * scale)
        assert torch.isfinite(out).all()


def test_v2_stacked_frames_equal_the_clip():
    model = _random(_small(TFDATv2, clip_size=3))
    frames = _clip(3)
    with torch.inference_mode():
        clip = model(frames)
        stacked = model(frames.reshape(1, 9, 27, 45))
    assert torch.equal(clip, stacked)
    with pytest.raises(ValueError, match="clip_size=3"):
        model(_clip(5))


@pytest.mark.parametrize("version", [TFDAT, TFDATv2])
def test_neighbours_change_the_output(version: Version):
    model = _random(_small(version))
    frames = _clip(5)
    moved = frames.clone()
    moved[:, 0] = moved[:, 0].roll(3, -1)
    with torch.inference_mode():
        assert not torch.equal(model(frames), model(moved))


@pytest.mark.parametrize("version", [TFDAT, TFDATv2])
def test_one_image_is_a_still_clip(version: Version):
    arch = TFDATArch() if version is TFDAT else TFDATv2Arch()
    descriptor = arch.load(_random(_small(version)).state_dict())
    image = _clip(1)[:, 0]
    with torch.inference_mode():
        out = descriptor(image)
        still = descriptor.model(image[:, None].repeat(1, 5, 1, 1, 1))
    assert out.shape == (1, 3, 54, 90)
    torch.testing.assert_close(out, still.clamp(0, 1), rtol=0, atol=0)


def test_tags_and_settings():
    descriptor = TFDATv2Arch().load(TFDATv2(clip_size=9, embed_dim=120).state_dict())
    assert descriptor.tags == ["120dim", "pixelshuffle", "clip 9", "temporal"]
    assert descriptor.scale == 2
    assert descriptor.model.clip_size == 9
    assert descriptor.model.align == 16
    assert descriptor.supports_bfloat16
    v1 = TFDATArch().load(TFDAT(scale=1).state_dict())
    assert v1.purpose == "Restoration"
    assert v1.model.align == 32
    assert v1.supports_bfloat16


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v1_warp_samples_in_fp32(dtype: torch.dtype):
    # 480 x 720: FP16 spaces coordinates past 512 by 0.5, and BF16 rounds
    # H - 1 = 479 up to 480, past the frame
    generator = torch.Generator().manual_seed(2)
    frame = torch.rand(1, 3, 480, 720, generator=generator).to(dtype)
    flow = (torch.rand(1, 2, 480, 720, generator=generator) * 8 - 4).to(dtype)
    warped = warp_frame(frame, flow)
    assert warped.dtype == dtype
    assert torch.equal(warped, warp_frame(frame.float(), flow.float()).to(dtype))
