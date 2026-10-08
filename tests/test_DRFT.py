import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.DRFT.__arch import drft_arch, legacy_drft_arch

from .util import skip_if_unchanged

skip_if_unchanged(__file__)


@pytest.mark.parametrize(
    ("make", "scale"),
    [
        (lambda: drft_arch.drft_nano(scale=4), 4),
        (lambda: drft_arch.drft_nano(scale=2), 2),
        (lambda: drft_arch.drft_micro(scale=4), 4),
        (
            lambda: legacy_drft_arch.DRFT(
                embed_dim=32, depths=(2, 2), num_heads=(2, 2), window_size=8, upscale=2
            ),
            2,
        ),
    ],
)
def test_loaded_model_matches(make, scale):
    torch.manual_seed(0)
    model = make().eval()
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert descriptor.architecture.id == "DRFT"
    assert descriptor.scale == scale

    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        expected = model(image)
        actual = descriptor.model(image)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
