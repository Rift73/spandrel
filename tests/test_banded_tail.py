from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from spandrel.architectures.__arch_helpers.inference import (
    banded_tail,
    tail_scale_and_halo,
)
from spandrel.architectures.ESRGAN import ESRGAN
from spandrel.architectures.ESRGAN.__arch import RRDB

# The tail's convolutions pick their algorithm by shape, so a band's float sums may be
# ordered differently from the whole image's: FP32 rounding, not a different computation.
ATOL = 1e-5

VARIANTS: dict[str, dict[str, Any]] = {
    "4x upconv": {"scale": 4},
    "2x upconv": {"scale": 2},
    "3x upconv": {"scale": 3},
    "8x upconv": {"scale": 8},
    "4x pixel shuffle": {"scale": 4, "upsampler": "pixel_shuffle"},
    "2x from pixel unshuffle (x2plus)": {"scale": 4, "shuffle_factor": 2, "in_nc": 12},
    "1x from pixel unshuffle": {"scale": 4, "shuffle_factor": 4, "in_nc": 48},
    "4x plus": {"scale": 4, "plus": True},
}


def esrgan(**settings) -> RRDB.RRDBNet:
    torch.manual_seed(0)
    net = ESRGAN(num_filters=16, num_blocks=1, **settings).eval()
    for p in net.parameters():
        nn.init.normal_(p, 0, 0.1)
    return net


def forced_bands(monkeypatch: pytest.MonkeyPatch, rows: int) -> None:
    """The banded path on the CPU, in bands of `rows` trunk rows."""
    monkeypatch.setattr(
        RRDB.RRDBNet,
        "tail_plan",
        lambda self, _x: tail_scale_and_halo(self.model[2:]),
    )
    monkeypatch.setattr(RRDB, "WHOLE_TAIL_ROWS", 0)
    monkeypatch.setattr(RRDB, "MIN_BAND_ROWS", rows)
    monkeypatch.setattr(RRDB, "BAND_DIVISOR", 10**6)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(("h", "w"), [(29, 17), (16, 16), (1, 9), (9, 1), (5, 5)])
def test_bands_compute_the_whole_pass(
    monkeypatch: pytest.MonkeyPatch, variant: str, h: int, w: int
):
    settings = VARIANTS[variant]
    if min(h, w) < settings.get("shuffle_factor", 1):
        pytest.skip("pixel unshuffle pads by reflection, which needs a wider image")
    net = esrgan(**settings)
    assert tail_scale_and_halo(net.model[2:]) is not None
    x = torch.rand(1, 3, h, w)
    with torch.no_grad():
        whole = net(x)
        forced_bands(monkeypatch, rows=3)
        banded = net(x)
    assert banded.shape == whole.shape
    torch.testing.assert_close(banded, whole, atol=ATOL, rtol=0)


def test_an_image_shorter_than_one_band_is_the_whole_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    net = esrgan(scale=4)
    calls: list[int] = []
    tail = net.model[2:]
    x = torch.rand(1, 16, 4, 6)
    with torch.no_grad():
        out = banded_tail(
            lambda t: calls.append(t.shape[2]) or tail(t), x, 4, 4, rows=8
        )
        torch.testing.assert_close(out, tail(x), atol=0, rtol=0)
    assert calls == [4]


def test_the_halo_counts_each_convolution_at_its_resolution():
    # 4x upconv: two convs after the 2x and 4x upsamplings (ceil(1/2), ceil(1/4)), HRconv
    # and conv_last at 4x: one row each of the tail's input
    assert tail_scale_and_halo(esrgan(scale=4).model[2:]) == (4, 4)
    assert tail_scale_and_halo(esrgan(scale=2).model[2:]) == (2, 3)
    assert tail_scale_and_halo(
        esrgan(scale=4, upsampler="pixel_shuffle").model[2:]
    ) == (4, 4)


@pytest.mark.parametrize(
    "layer",
    [
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(4, 4, 3, stride=2, padding=1),
        nn.Conv2d(4, 4, 3, padding=1, padding_mode="reflect"),
        nn.BatchNorm2d(4),
    ],
)
def test_tails_it_cannot_band_keep_the_whole_pass(layer: nn.Module):
    assert (
        tail_scale_and_halo(nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), layer)) is None
    )


def test_the_cpu_and_tracing_keep_the_whole_pass():
    net = esrgan(scale=4)
    with torch.no_grad():
        assert net.tail_plan(torch.rand(1, 16, 8, 8)) is None


def test_a_model_without_upscaling_keeps_the_whole_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    net = esrgan(scale=1)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.version, "hip", None)
    with torch.no_grad():
        assert net.tail_plan(torch.rand(1, 16, 8, 8)) is None
        assert esrgan(scale=4).tail_plan(torch.rand(1, 16, 8, 8)) == (4, 4)
        monkeypatch.setattr(torch.version, "hip", "7.0")
        assert esrgan(scale=4).tail_plan(torch.rand(1, 16, 8, 8)) is None
