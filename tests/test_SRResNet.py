import torch
from spandrel.architectures.SRResNet import MSRResNet, SRResNet, SRResNetArch, SRResNetSequential

from .util import (
    assert_loads_correctly,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        SRResNetArch(),
        lambda: MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4),
        lambda: MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=8, upscale=2),
        lambda: SRResNet(in_nc=3, out_nc=3, nf=64, nb=16, upscale=4),
        lambda: SRResNet(in_nc=3, out_nc=3, nf=32, nb=8, upscale=2, norm="batch"),
        lambda: SRResNetSequential(in_nc=3, out_nc=3, nf=64, nb=16, scale=2),
    )


def test_inference():
    model = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=4, upscale=2).eval()
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 64, 64)
