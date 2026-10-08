import importlib

import pytest
import torch

# Channels-first LayerNorms that compute their variance by hand. In fp16,
# (x - mean)^2 overflowed once |x - mean| exceeded 256, which zeroed the
# normalized values (upstream chaiNNer #3071).
LAYER_NORMS = [
    ("OmniSR.__arch.layernorm", "LayerNorm2d", {}),
    ("NAFNet.__arch.arch_util", "LayerNorm2d", {}),
    ("KBNet.__arch.kb_utils", "LayerNorm2d", {}),
    ("FDAT.__arch.fdat", "LayerNorm", {}),
    ("HVICIDNet.__arch.transformer_utils", "LayerNorm", {}),
    ("MoESR.__arch.MoESR", "LayerNorm", {}),
    ("MoSR.__arch.mosr_arch", "LayerNorm", {}),
    ("PLKSR.__arch.RealPLKSR", "LayerNorm", {}),
    ("SAFMN.__arch.safmn", "LayerNorm", {}),
    ("SAFMNBCIE.__arch.safmn_bcie", "LayerNorm", {}),
    ("SeemoRe.__arch.seemore_arch", "LayerNorm", {"data_format": "channels_first"}),
]


@pytest.mark.parametrize(("module", "name", "kwargs"), LAYER_NORMS)
def test_fp16_statistics_do_not_overflow(module: str, name: str, kwargs: dict):
    cls = getattr(importlib.import_module("spandrel.architectures." + module), name)
    norm = cls(8, **kwargs).eval()

    x = torch.randn(1, 8, 4, 4)
    x[0, :, 1, 2] = torch.tensor([300.0, -300.0] * 4)

    with torch.no_grad():
        expected = norm(x)
        actual = norm.half()(x.half())

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=1e-2)
