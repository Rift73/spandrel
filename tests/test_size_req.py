from unittest import mock

import pytest
import torch

from spandrel import SizeRequirements
from spandrel.__helpers.size_req import pad_tensor


def test_size_req_init():
    a = SizeRequirements(minimum=1, multiple_of=2)
    assert a.minimum == 2
    assert a.multiple_of == 2

    with pytest.raises(AssertionError):
        SizeRequirements(minimum=-1)
    with pytest.raises(AssertionError):
        SizeRequirements(multiple_of=0)
    with pytest.raises(AssertionError):
        SizeRequirements(multiple_of=-1)


def _trace_pad_tensor(req: SizeRequirements) -> torch.jit.ScriptModule:
    class Pad(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return pad_tensor(x, req)[1]

    # torch.onnx.export traces like this, with ONNX export mode on
    with mock.patch.object(torch.onnx, "is_in_onnx_export", return_value=True):
        return torch.jit.trace(Pad(), torch.rand(1, 3, 32, 32), check_trace=False)


@pytest.mark.parametrize(
    "req",
    [
        SizeRequirements(multiple_of=8),
        SizeRequirements(minimum=16, multiple_of=8),
        SizeRequirements(minimum=40, multiple_of=4),
        SizeRequirements(multiple_of=4, square=True),
    ],
)
def test_pad_tensor_traced(req: SizeRequirements):
    # upstream chaiNNer #1816: the traced graph froze the example input's padding
    traced = _trace_pad_tensor(req)
    for height, width in [(1, 1), (2, 3), (5, 7), (16, 16), (17, 31), (37, 51)]:
        x = torch.rand(1, 3, height, width)
        assert torch.equal(traced(x), pad_tensor(x, req)[1]), (width, height)


def test_pad_tensor_traced_minimum_only():
    # a minimum alone adds no padding to the graph (e.g. plain ESRGAN's, which
    # NCNN's converter could not take with shape operations in it)
    traced = _trace_pad_tensor(SizeRequirements(minimum=16))
    assert not [n for n in traced.graph.nodes() if "pad" in n.kind()]
