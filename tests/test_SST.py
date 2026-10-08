import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.SST.__arch import sst_arch

from .util import skip_if_unchanged

skip_if_unchanged(__file__)


def test_loaded_model_matches():
    torch.manual_seed(0)
    model = sst_arch.sst_light(scale=4, attn_type="RIB").eval()
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert descriptor.architecture.id == "SST"
    assert descriptor.scale == 4

    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(image)
        actual = descriptor.model(image)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_sst_real_loads_as_a_single_step_model():
    torch.manual_seed(0)
    model = sst_arch.sst_real(scale=4)
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert descriptor.architecture.id == "SSTReal"
    assert descriptor.scale == 4

    with torch.no_grad():
        output = descriptor.model.eval()(torch.rand(1, 3, 64, 64))
    assert output.shape == (1, 3, 256, 256)
