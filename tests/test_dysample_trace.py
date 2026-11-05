import pytest
import torch
from torch import nn

from spandrel.architectures.__arch_helpers.dysample import DySample
from spandrel.architectures.FDAT.__arch.fdat import DySample as FdatDySample
from spandrel.architectures.GaterV3.__arch.gaterv3 import DySample as GaterDySample
from spandrel.architectures.GFISRv2.__arch.gfisrv2 import DySample as GfisrDySample
from spandrel.architectures.SMoSR.__arch.smosr_arch import DySample as SmosrDySample


@pytest.mark.parametrize(
    "dysample", [DySample, FdatDySample, GaterDySample, GfisrDySample, SmosrDySample]
)
def test_a_trace_keeps_the_sampling_grid_dynamic(dysample: type[nn.Module]):
    # An ONNX export traces one example size; its grid must follow the input's size.
    model = dysample(16, 3, 2, 4).eval().requires_grad_(False)
    traced = torch.jit.trace(model, torch.rand(1, 16, 32, 32))
    image = torch.rand(1, 16, 37, 51)
    torch.testing.assert_close(traced(image), model(image), rtol=0, atol=1e-5)
