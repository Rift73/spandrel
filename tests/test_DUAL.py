import copy
import math

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.DUAL import DUAL, DUALArch
from spandrel.architectures.DUAL.__arch import dual_arch
from spandrel.architectures.DUAL.__arch.dual_arch import (
    _SpatialAttention,
    dual_light,
    dual_xs,
)

from .util import assert_loads_correctly, assert_size_requirements, skip_if_unchanged

skip_if_unchanged(__file__)


def _small(**kwargs):
    return DUAL(
        embed_dim=32,
        depths=(4, 4),
        layer_indices=((0, 1, 2, 5), (2, 3, 4, 5)),
        **kwargs,
    )


def test_load():
    assert_loads_correctly(
        DUALArch(),
        lambda: dual_light(4),
        lambda: dual_xs(4),
        lambda: dual_light(2),
        lambda: dual_xs(1),
        lambda: DUAL(embed_dim=32, depths=(6, 6)),
        lambda: DUAL(scale=2, embed_dim=64, depths=(6, 6, 6)),
        lambda: _small(scale=1),
        lambda: _small(scale=2, pixel_unshuffle=True),
        lambda: _small(c_pos_bias="rib", context_na=False),
        lambda: _small(c_pos_bias="rib", rfb_rank=8, context_na=False),
        lambda: _small(c_pos_bias="none", context_na=False),
        lambda: _small(rfb_rank=8),
        lambda: _small(qk="cosine", context_na=False),
        lambda: _small(g_window=8),
        lambda: _small(dynamic_experts=2),
        lambda: DUAL(embed_dim=32, depths=(2, 4), layer_indices=((0, 1), (2, 3, 4, 5))),
        # the factories leave context_na at None (meaning True), the loader sets
        # it explicitly; test_context_na checks the resolved value
        ignore_parameters={"context_na"},
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),
        ({"c_pos_bias": "rib", "context_na": False}, False),
        ({"c_pos_bias": "none", "context_na": False}, False),
        ({"qk": "cosine", "context_na": False}, False),
    ],
)
def test_context_na(kwargs, expected):
    model = _small(**kwargs)
    loaded = DUALArch().load(model.state_dict()).model
    for m in (model, loaded):
        assert m.context_na is expected
        # (window, context_na) of every layer: local G16 layers never route NA
        routes = {
            (a.window, a.context_na)
            for a in m.modules()
            if isinstance(a, _SpatialAttention)
        }
        assert routes == {(16, False), (32, expected), (64, expected)}
    if not expected:
        # neighborhood attention is only defined for RFB bias and dot attention
        with pytest.raises(ValueError, match="context_na requires"):
            _small(**{k: v for k, v in kwargs.items() if k != "context_na"})


def _na_reference(attention, x):
    """Dense FP64 attention in which every query sees only its own clamped
    neighborhood, with the RFB bias of the query-minus-key offset."""
    batch, channels, height, width = x.shape
    heads, dim = attention.heads, channels // attention.heads
    kh, kw = min(height, attention.window), min(width, attention.window)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    y, z = yy.flatten(), xx.flatten()
    sy = (y - kh // 2).clamp(0, height - kh)
    sz = (z - kw // 2).clamp(0, width - kw)
    inside = (
        (y >= sy[:, None])
        & (y < sy[:, None] + kh)
        & (z >= sz[:, None])
        & (z < sz[:, None] + kw)
    )
    rfb = attention.factored_bias
    offset = torch.stack((y[:, None] - y, z[:, None] - z), dim=-1).double()
    phase = offset @ rfb.frequency.T
    bias = torch.einsum("hr,qkr->hqk", rfb.amplitude_cos, phase.cos())
    bias = bias + torch.einsum("hr,qkr->hqk", rfb.amplitude_sin, phase.sin())
    tokens = x.flatten(2).transpose(1, 2)
    qkv = attention.qkv(tokens).reshape(batch, -1, 3, heads, dim)
    query, key, value = qkv.permute(2, 0, 3, 1, 4)
    logits = query @ key.transpose(-1, -2) * dim**-0.5 + bias
    weights = logits.masked_fill(~inside, -math.inf).softmax(-1)
    return (weights @ value).transpose(-1, -2).reshape(batch, channels, height, width)


# odd and rectangular inputs, one axis at or below the window, partial 16x16
# query tiles, a window above the tile size; with autograd (training chunks)
# and without (inference bands)
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize(
    ("window", "shape"), [(6, (13, 21)), (7, (5, 37)), (6, (34, 6)), (20, (45, 23))]
)
def test_neighborhood_attention_matches_dense_reference(window, shape, grad):
    torch.manual_seed(0)
    attention = _SpatialAttention(8, 2, window, rank=4, context_na=True).double()
    attention.eval()
    for name, parameter in attention.named_parameters():
        if name.startswith("factored_bias."):
            parameter.data.normal_(0, 0.5)
    x = torch.randn(2, 8, *shape, dtype=torch.float64)
    with torch.set_grad_enabled(grad):
        actual = attention.pre_gate_core(x)
    with torch.no_grad():
        expected = _na_reference(attention, x)
    torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    ("cap", "calls"),
    [
        # 15 x 10 query tiles, batch 2: bands of 6 whole rows
        (dual_arch.NA_INFERENCE_MASK_ELEMENTS, 3),
        # a row above the cap: chunks of 4 tiles within the row
        (4 * 16 * 16 * 63 * 63 * 2, 38),
    ],
)
def test_neighborhood_attention_inference_chunks(monkeypatch, cap, calls):
    torch.manual_seed(0)
    attention = _SpatialAttention(8, 2, 48, rank=4, context_na=True).eval()
    x = torch.randn(2, 8, 240, 160)
    training = attention.pre_gate_core(x).detach()
    monkeypatch.setattr(dual_arch, "NA_INFERENCE_MASK_ELEMENTS", cap)
    sdpa = dual_arch.F.scaled_dot_product_attention
    counted = []
    monkeypatch.setattr(
        dual_arch.F,
        "scaled_dot_product_attention",
        lambda *a, **k: counted.append(1) or sdpa(*a, **k),
    )
    with torch.no_grad():
        inference = attention.pre_gate_core(x)
    assert len(counted) == calls
    torch.testing.assert_close(inference, training, rtol=0, atol=0)


def test_neighborhood_attention_dispatch(monkeypatch):
    attention = _SpatialAttention(8, 2, 6, rank=4, context_na=True)
    shapes = []
    monkeypatch.setattr(
        attention, "_na_core", lambda x: shapes.append(tuple(x.shape[2:])) or x
    )
    with torch.no_grad():
        for shape in ((6, 6), (4, 6), (6, 1), (7, 6), (6, 7), (3, 40)):
            attention.pre_gate_core(torch.rand(1, 8, *shape))
    # an input within the window on both axes keeps the whole-window path
    assert shapes == [(7, 6), (6, 7), (3, 40)]


def test_ambiguous_schedule_is_refused():
    # (3, 4, 5) and (0, 1, 2) both put the context layer last
    model = DUAL(embed_dim=32, depths=(3, 3), layer_indices=((0, 1, 2), (3, 4, 5)))
    with pytest.raises(ValueError, match="layer schedule"):
        DUALArch().load(model.state_dict())


@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize("pixel_unshuffle", [False, True])
def test_loaded_model_matches(folded, pixel_unshuffle):
    torch.manual_seed(0)
    model = _small(scale=2, pixel_unshuffle=pixel_unshuffle).eval()
    # the expert kernels start at zero; give the routed path some weight
    for name, parameter in model.named_parameters():
        if name.endswith("conv.experts"):
            parameter.data.normal_(0, 0.02)
    source = copy.deepcopy(model)
    if folded:
        source.prepare_for_export()
    descriptor = MAIN_REGISTRY.load(source.state_dict())
    assert descriptor.architecture.id == "DUAL"
    assert descriptor.scale == 2

    image = torch.rand(1, 3, 37, 29)
    with torch.no_grad():
        expected = model(image)
        actual = descriptor.model(image)
    assert actual.shape == (1, 3, 74, 58)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("groups", [2, 3])
def test_half_survives_a_last_group_beyond_fp16(groups):
    torch.manual_seed(0)
    indices = ((0, 1, 2, 5), (2, 3, 4, 5), (0, 1, 2, 5))[:groups]
    model = DUAL(scale=2, embed_dim=32, depths=(4,) * groups, layer_indices=indices)
    model.eval().prepare_for_export()
    for name, parameter in model.named_parameters():
        if name.endswith("conv.experts"):
            parameter.data.normal_(0, 0.02)
    # A trained 1x Light's last group conv writes about 125k from inputs of a few
    # hundred, beyond FP16's 65504; the final norm right after it scales that back
    # down. Here: inputs of about 50, outputs of about 250k, weights below 1000.
    group = model.groups[-1]
    project = group.layers[-1].ffn.project
    for parameter in (project.weight, project.bias):
        parameter.data.mul_(1000)
    for parameter in (group.conv.base.weight, group.conv.base.bias, group.conv.experts):
        parameter.data.mul_(1e4)
    # Two images: a batch takes the dynamic conv's grouped path.
    image = torch.rand(2, 3, 37, 29)
    with torch.no_grad():
        expected = model(image)
        assert not torch.isfinite(copy.deepcopy(model).half()(image.half())).all()

    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert descriptor.supports_half
    assert descriptor.model.state_dict().keys() == model.state_dict().keys()
    with torch.no_grad():
        actual = descriptor.model.half()(image.half()).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=5e-3)


def test_size_requirements():
    assert_size_requirements(DUALArch().load(_small(scale=2).state_dict()))
    assert_size_requirements(DUALArch().load(_small(pixel_unshuffle=True).state_dict()))
