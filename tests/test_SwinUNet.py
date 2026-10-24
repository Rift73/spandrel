import torch
from spandrel.architectures.SwinUNet import SwinUNet, SwinUNetArch, SwinUNetBase

from .util import (
    assert_loads_correctly,
    skip_if_unchanged,
)

skip_if_unchanged(__file__)


def test_load():
    assert_loads_correctly(
        SwinUNetArch(),
        lambda: SwinUNet(
            SwinUNetBase(in_channels=3, out_channels=3, base_dim=96, base_layers=2, scale_factor=2),
            scale=2,
        ),
        lambda: SwinUNet(
            SwinUNetBase(in_channels=3, out_channels=3, base_dim=96, base_layers=2, scale_factor=1),
            scale=1,
        ),
        lambda: SwinUNet(
            SwinUNetBase(in_channels=3, out_channels=3, base_dim=96, base_layers=2, scale_factor=4),
            scale=4,
        ),
    )


def test_inference():
    model = SwinUNet(
        SwinUNetBase(in_channels=3, out_channels=3, base_dim=96, base_layers=2, scale_factor=2),
        scale=2,
    ).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 128, 128)
