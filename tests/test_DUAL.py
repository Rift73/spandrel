import copy
import math

import pytest
import torch

from spandrel import MAIN_REGISTRY
from spandrel.architectures.DUAL import DUAL, DUALArch
from spandrel.architectures.DUAL.__arch import dual_arch, na_local_triton
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
        lambda: _small(qk="cosine", context_na=False, local_na=False),
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
        ({"qk": "cosine", "context_na": False, "local_na": False}, False),
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
# query tiles, a window above the tile size, clear tiles sharing a mask; with
# autograd (training chunks) and without (inference calls)
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize(
    ("window", "shape"),
    [(6, (13, 21)), (7, (5, 37)), (6, (34, 6)), (20, (45, 23)), (6, (45, 50))],
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
        # 15 x 10 query tiles, batch 2: up to 66 a call, 84 edge and 66 clear ones
        (dual_arch.NA_INFERENCE_MASK_ELEMENTS, 3),
        # 4 a call
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


def _regional_class(rca, q, k, v, sy, sx):
    """One class padding its own copies and working out its own geometry."""
    b, c, h, w = q.shape
    heads, d = rca.heads, c // rca.heads
    ph, pw = h + sy + (-(h + sy) % 32), w + sx + (-(w + sx) % 32)
    ny, nx = ph // 32, pw // 32

    def windows(t):
        return (
            torch.nn.functional.pad(t, (sx, pw - w - sx, sy, ph - h - sy))
            .reshape(b, heads, d, ny, 32, nx, 32)
            .permute(0, 3, 5, 1, 2, 4, 6)
            .reshape(b, ny * nx, heads, d, 1024)
        )

    y0, x0 = torch.arange(ny) * 32 - sy, torch.arange(nx) * 32 - sx
    yc = (y0 + 32).clamp(max=h) - y0.clamp(min=0)
    xc = (x0 + 32).clamp(max=w) - x0.clamp(min=0)
    count = (yc[:, None] * xc[None, :]).reshape(1, ny * nx, 1, 1)
    a = rca.channel_attention(windows(q), windows(k), count)
    o = (
        (a @ windows(v))
        .reshape(b, ny, nx, heads, d, 32, 32)
        .permute(0, 3, 4, 1, 5, 2, 6)
        .reshape(b, c, ph, pw)[:, :, sy : sy + h, sx : sx + w]
    )
    wy = dual_arch.hat_weights(h, sy)[None, None, :, None]
    wx = dual_arch.hat_weights(w, sx)[None, None, None, :]
    return o * wy * wx


# every residue of the two axes against the 32-pixel grids of the shifted classes;
# with autograd (geometry computed) and without (geometry reused)
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("shape", [(33, 47), (64, 32), (16, 48), (5, 7), (48, 16)])
def test_regional_mix_matches_per_class_padding(shape, grad):
    torch.manual_seed(0)
    rca = dual_arch.RegionalChannelAttention(64, 2).eval()
    q, k, v = torch.randn(3, 1, 64, *shape).unbind(0)
    with torch.set_grad_enabled(grad):
        actual = rca.mix(q, k, v)
    with torch.no_grad():
        expected = (
            _regional_class(rca, q, k, v, 0, 0)
            + _regional_class(rca, q, k, v, 0, 16)
            + _regional_class(rca, q, k, v, 16, 0)
            + _regional_class(rca, q, k, v, 16, 16)
        )
    # inference sums the windows' products from their cells, which rounds its own way
    atol = 0 if grad else 1e-5
    torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=atol)


# A convolution's rounding depends on the layout of its input; layers run channels-last
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_group_convolutions_see_the_input_layout(layout):
    model = _small(scale=2).eval()
    model.prepare_for_export()
    seen = []
    for module in model.modules():
        if isinstance(module, dual_arch.PremixedDeployedDynamicConv3x3):
            module.register_forward_pre_hook(
                lambda _, inputs: seen.append(
                    inputs[0].is_contiguous(memory_format=layout)
                )
            )
    with torch.no_grad():
        model(torch.rand(1, 3, 37, 29).contiguous(memory_format=layout))
    assert seen == [True, True]


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


def _local_na_reference(attention, x):
    """Dense FP64 attention in which every query sees only its own clamped
    window x window neighborhood, with the dense RPB of the query-minus-key offset."""
    batch, channels, height, width = x.shape
    heads, dim, window = attention.heads, channels // attention.heads, attention.window
    kh, kw = min(height, window), min(width, window)
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
    span = 2 * window - 1
    dy = (y[:, None] - y + window - 1).clamp(0, span - 1)
    dz = (z[:, None] - z + window - 1).clamp(0, span - 1)
    bias = attention.relative_bias[dy * span + dz].permute(2, 0, 1)
    tokens = x.flatten(2).transpose(1, 2)
    qkv = attention.qkv(tokens).reshape(batch, -1, 3, heads, dim)
    query, key, value = qkv.permute(2, 0, 3, 1, 4)
    logits = query @ key.transpose(-1, -2) * dim**-0.5 + bias
    weights = logits.masked_fill(~inside, -math.inf).softmax(-1)
    return (weights @ value).transpose(-1, -2).reshape(batch, channels, height, width)


# above the window on one or both axes, odd sizes, partial 8x8 query tiles, batch 2
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize(
    ("window", "shape"), [(16, (17, 30)), (16, (40, 9)), (6, (13, 21)), (8, (25, 25))]
)
def test_local_na_matches_dense_reference(window, shape, grad):
    torch.manual_seed(0)
    attention = _SpatialAttention(8, 2, window, pos_bias="rpb").double().eval()
    assert attention.relative_bias is not None
    attention.relative_bias.data.normal_(0, 0.5)
    attention.local_na = True
    x = torch.randn(2, 8, *shape, dtype=torch.float64)
    with torch.set_grad_enabled(grad):
        actual = attention.pre_gate_core(x)
    with torch.no_grad():
        expected = _local_na_reference(attention, x)
    torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=1e-10)


def test_local_na_keeps_the_window_path_up_to_the_window():
    torch.manual_seed(0)
    attention = _SpatialAttention(8, 2, 16, pos_bias="rpb").eval()
    x = torch.randn(1, 8, 16, 12)
    with torch.no_grad():
        expected = attention.pre_gate_core(x)
        attention.local_na = True
        assert torch.equal(attention.pre_gate_core(x), expected)


def _retrieval_reference(q, k, v, log_tau, window=48):
    """Dense FP64 retrieval: fine query (y, x) attends the window x window coarse keys
    around (y // 2, x // 2), clamped inside the coarse grid."""
    batch, channels, height, width = q.shape
    heads = channels // 32
    kh_, kw_ = k.shape[-2:]
    kh, kw = min(window, kh_), min(window, kw_)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    y, z = yy.flatten(), xx.flatten()
    ky, kx = torch.meshgrid(torch.arange(kh_), torch.arange(kw_), indexing="ij")
    ky, kx = ky.flatten(), kx.flatten()
    sy = (y // 2 - kh // 2).clamp(0, kh_ - kh)
    sz = (z // 2 - kw // 2).clamp(0, kw_ - kw)
    inside = (
        (ky >= sy[:, None])
        & (ky < sy[:, None] + kh)
        & (kx >= sz[:, None])
        & (kx < sz[:, None] + kw)
    )
    tau = log_tau.exp().clamp(0.1, 10)
    query = q.reshape(batch, heads, 32, -1).transpose(-1, -2) * tau[None, :, None, None]
    key = k.reshape(batch, heads, 32, -1).transpose(-1, -2)
    value = v.reshape(batch, heads, 32, -1).transpose(-1, -2)
    scale = math.log(kh * kw) / (math.log(1024) * math.sqrt(32))
    logits = (query @ key.transpose(-1, -2) * scale).masked_fill(~inside, -math.inf)
    out = logits.softmax(-1) @ value
    return out.transpose(-1, -2).reshape(batch, channels, height, width)


@pytest.mark.parametrize("coarse", [(49, 30), (30, 52), (50, 50)])
def test_retrieval_na_matches_dense_reference(coarse):
    torch.manual_seed(0)
    ch, cw = coarse
    q = torch.randn(2, 64, 2 * ch, 2 * cw, dtype=torch.float64)
    k = torch.randn(2, 64, ch, cw, dtype=torch.float64)
    v = torch.randn(2, 64, ch, cw, dtype=torch.float64)
    log_tau = torch.randn(2, dtype=torch.float64) * 0.3
    packed = dual_arch.pack_qk(q, k, na=True)
    assert isinstance(packed[0], dual_arch.NAGeometry)
    actual = dual_arch.attend_packed(packed, v, log_tau)
    expected = _retrieval_reference(q, k, v, log_tau)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


def test_retrieval_na_keeps_the_blocks_up_to_48():
    q = torch.randn(1, 64, 96, 90)
    k = torch.randn(1, 64, 48, 45)
    blocks = dual_arch.pack_qk(q, k)
    packed = dual_arch.pack_qk(q, k, na=True)
    assert not isinstance(packed[0], dual_arch.NAGeometry)
    v = torch.randn(1, 64 * 4, 48, 45)
    log_tau = torch.zeros(2)
    assert torch.equal(
        dual_arch.attend_packed(packed, v, log_tau),
        dual_arch.attend_packed(blocks, v, log_tau),
    )


def test_every_dot_dual_loads_with_local_and_retrieval_na():
    cosine = _small(qk="cosine", context_na=False, local_na=False)
    for source, local in ((dual_light(2), True), (cosine, False)):
        model = DUALArch().load(source.state_dict()).model
        assert model.local_na is local
        assert model.retrieval_na
        layers = [
            a.local_na
            for a in model.modules()
            if isinstance(a, _SpatialAttention) and a.pos_bias == "rpb"
        ]
        assert layers and all(layer is local for layer in layers)
    with pytest.raises(ValueError, match="local_na requires"):
        _small(qk="cosine", context_na=False)


needs_kernel = pytest.mark.skipif(
    not (
        na_local_triton.AVAILABLE
        and torch.cuda.is_available()
        and torch.version.hip is None
    ),
    reason="the kernels run on NVIDIA GPUs with Triton",
)


def _counted(monkeypatch, name):
    calls = []
    launch = getattr(na_local_triton, name)

    def counted(*args):
        calls.append(args)
        return launch(*args)

    monkeypatch.setattr(na_local_triton, name, counted)
    return calls


def _local_attention(channels, window):
    attention = _SpatialAttention(channels, channels // 32, window, pos_bias="rpb")
    assert attention.relative_bias is not None
    attention.relative_bias.data.normal_(0, 0.5)
    attention.local_na = True
    return attention.cuda().eval().half()


# batch 2, non-square, partial 8x8 query tiles, above the window on one or both axes
@needs_kernel
@pytest.mark.parametrize(
    ("batch", "channels", "window", "shape"),
    [
        (2, 128, 16, (40, 37)),
        (1, 128, 16, (17, 200)),
        (1, 64, 16, (96, 9)),
        (2, 64, 8, (25, 13)),
        (1, 64, 16, (5, 40)),
    ],
)
def test_local_na_kernel_matches_the_pytorch_form(
    monkeypatch, batch, channels, window, shape
):
    torch.manual_seed(0)
    attention = _local_attention(channels, window)
    x = torch.randn(batch, channels, *shape, device="cuda").half()
    calls = _counted(monkeypatch, "local_na")
    with torch.no_grad():
        actual = attention.pre_gate_core(x)
        assert len(calls) == 1
        monkeypatch.setattr(dual_arch, "LOCAL_NA_TRITON", False)
        expected = attention.pre_gate_core(x)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-3)


@needs_kernel
def test_local_na_kernel_stays_off_up_to_the_window_and_outside_inference(
    monkeypatch,
):
    attention = _local_attention(64, 16)
    calls = _counted(monkeypatch, "local_na")
    with torch.no_grad():
        attention.pre_gate_core(torch.randn(1, 64, 16, 16, device="cuda").half())
        attention.float().pre_gate_core(torch.randn(1, 64, 17, 17, device="cuda"))
    attention.half().pre_gate_core(torch.randn(1, 64, 17, 17, device="cuda").half())
    with torch.no_grad():
        attention.bfloat16().pre_gate_core(
            torch.randn(1, 64, 17, 17, device="cuda").bfloat16()
        )
        attention.half()
        with torch.autocast("cuda", torch.float16):
            attention.pre_gate_core(torch.randn(1, 64, 17, 17, device="cuda").half())
    assert not calls


def _retrieval_operands(batch, channels, fine, coarse):
    q = torch.randn(batch, channels, *fine, device="cuda").half()
    k = torch.randn(batch, channels, *coarse, device="cuda").half()
    v = torch.randn(batch, channels, *coarse, device="cuda").half()
    log_tau = (torch.randn(channels // 32, device="cuda") * 0.3).half()
    return q, k, v, log_tau


# coarse grids above 48 on one or both axes, odd fine sizes, batch 2
@needs_kernel
@pytest.mark.parametrize(
    ("batch", "channels", "fine", "coarse"),
    [
        (2, 128, (100, 98), (50, 49)),
        (1, 64, (120, 60), (60, 30)),
        (2, 64, (97, 41), (49, 21)),
        (1, 64, (2, 120), (1, 60)),
        (2, 64, (6, 99), (3, 50)),
    ],
)
def test_retrieval_na_kernel_matches_the_pytorch_form(
    monkeypatch, batch, channels, fine, coarse
):
    torch.manual_seed(0)
    q, k, v, log_tau = _retrieval_operands(batch, channels, fine, coarse)
    calls = _counted(monkeypatch, "retrieval_na")
    with torch.no_grad():
        actual = dual_arch.attend_packed(dual_arch.pack_qk(q, k, na=True), v, log_tau)
        assert len(calls) == 1
        monkeypatch.setattr(dual_arch, "RETRIEVAL_NA_TRITON", False)
        expected = dual_arch.attend_packed(dual_arch.pack_qk(q, k, na=True), v, log_tau)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-3)


@needs_kernel
def test_retrieval_na_kernel_stays_off_up_to_48(monkeypatch):
    q, k, v, log_tau = _retrieval_operands(1, 64, (96, 90), (48, 45))
    calls = _counted(monkeypatch, "retrieval_na")
    with torch.no_grad():
        dual_arch.attend_packed(dual_arch.pack_qk(q, k, na=True), v, log_tau)
    assert not calls


@needs_kernel
def test_retrieval_na_kernel_stays_off_outside_fp16_cuda_inference(monkeypatch):
    q, k, v, log_tau = _retrieval_operands(1, 64, (100, 98), (50, 49))
    calls = _counted(monkeypatch, "retrieval_na")

    def attend(*operands):
        q, k, v, log_tau = operands
        return dual_arch.attend_packed(dual_arch.pack_qk(q, k, na=True), v, log_tau)

    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            attend(*(t.to(dtype) for t in (q, k, v, log_tau)))
        attend(*(t.float().cpu() for t in (q, k, v, log_tau)))
        with torch.autocast("cuda", torch.float16):
            attend(q, k, v, log_tau)
    attend(q, k, v, log_tau)
    assert not calls


# 2,056 x 2,056 queries are 66,049 tiles of 8x8, past the 65,535 programs a second
# grid axis holds; one head keeps the tensors as small as that grid allows. The
# queries checked are the last tile's (the programs past 65,535) and the corners.
_GRID_SIDE = 2056


def _grid_queries():
    last = torch.arange(_GRID_SIDE - 8, _GRID_SIDE)
    ys = torch.cat([last.repeat_interleave(8), torch.tensor([0, 0, 1031, 2055])])
    xs = torch.cat([last.repeat(8), torch.tensor([0, 2055, 517, 0])])
    return ys.cuda(), xs.cuda()


def _neighborhoods(ys, xs, side, size):
    """Row-major key positions of each query's clamped size x size neighborhood."""
    offsets = torch.arange(size, device=ys.device)
    starts = [(t - size // 2).clamp(0, side - size) for t in (ys, xs)]
    ky = (starts[0][:, None] + offsets).repeat_interleave(size, 1)
    kx = (starts[1][:, None] + offsets).repeat(1, size)
    return ky, kx


@needs_kernel
def test_local_na_kernel_covers_more_tiles_than_a_grid_axis_holds():
    torch.manual_seed(0)
    side, window = _GRID_SIDE, 16
    qkv = torch.randn(1, side * side, 96, device="cuda").half()
    table = (torch.randn(1, (2 * window - 1) ** 2, device="cuda") * 0.5).half()
    with torch.no_grad():
        out = na_local_triton.local_na(qkv, table, side, side, window)
    ys, xs = _grid_queries()
    ky, kx = _neighborhoods(ys, xs, side, window)
    tokens = qkv[0].float()
    q = (tokens[ys * side + xs, :32] * 32**-0.5).half().float()
    keys = tokens[ky * side + kx]
    dy, dx = ys[:, None] - ky + window - 1, xs[:, None] - kx + window - 1
    bias = table[0].float()[dy * (2 * window - 1) + dx]
    logits = (keys[..., 32:64] @ q[:, :, None])[..., 0] + bias
    expected = (logits.softmax(-1)[..., None] * keys[..., 64:]).sum(1)
    torch.testing.assert_close(out[0, ys, xs].float(), expected, rtol=0, atol=2e-3)


@needs_kernel
def test_retrieval_na_kernel_covers_more_tiles_than_a_grid_axis_holds():
    torch.manual_seed(0)
    side, coarse, window = _GRID_SIDE, _GRID_SIDE // 2, 48
    q = torch.randn(1, 32, side, side, device="cuda").half()
    k = torch.randn(1, 32, coarse, coarse, device="cuda").half()
    v = torch.randn(1, 32, coarse, coarse, device="cuda").half()
    tau = torch.tensor([1.5], device="cuda").half()
    with torch.no_grad():
        out = na_local_triton.retrieval_na(q, k, v, tau)
    ys, xs = _grid_queries()
    ky, kx = _neighborhoods(ys // 2, xs // 2, coarse, window)
    query = (q[0, :, ys, xs].float().t() * tau.float()).half().float()
    keys, values = (t[0, :, ky, kx].float().permute(1, 2, 0) for t in (k, v))
    scale = math.log(window * window) / (math.log(1024) * math.sqrt(32))
    logits = (keys @ query[:, :, None])[..., 0] * scale
    expected = (logits.softmax(-1)[..., None] * values).sum(1)
    torch.testing.assert_close(
        out[0, :, ys, xs].float().t(), expected, rtol=0, atol=2e-3
    )


# a 4x model on a 104 px input: local layers above 16 positions and a 52-wide coarse
# grid run both kernels, the retrieval once and again in the second-octave transfer
@needs_kernel
def test_a_4x_forward_with_the_kernels_matches_the_pytorch_form(monkeypatch):
    torch.manual_seed(0)
    model = _small(scale=4).cuda().eval()
    x = torch.rand(1, 3, 104, 96, device="cuda")
    local, retrieval = (
        _counted(monkeypatch, "local_na"),
        _counted(monkeypatch, "retrieval_na"),
    )
    with torch.no_grad():
        reference = model(x)
        model.half()
        assert not local and not retrieval
        actual = model(x.half()).float()
        assert local and len(retrieval) == 2
        monkeypatch.setattr(dual_arch, "LOCAL_NA_TRITON", False)
        monkeypatch.setattr(dual_arch, "RETRIEVAL_NA_TRITON", False)
        expected = model(x.half()).float()
    assert actual.shape == (1, 3, 416, 384)
    kernel_error = (actual - reference).square().mean().sqrt()
    pytorch_error = (expected - reference).square().mean().sqrt()
    assert kernel_error <= pytorch_error * 1.1 + 1e-6


class _TritonError(Exception):
    pass


_TritonError.__module__ = "triton.compiler.errors"


def _local_na_failing_with(monkeypatch, error):
    def launch(*_):
        raise error

    monkeypatch.setattr(dual_arch, "LOCAL_NA_TRITON", True)
    monkeypatch.setattr(dual_arch, "_na_kernel_usable", lambda *_: True)
    monkeypatch.setattr(na_local_triton, "local_na", launch)
    torch.manual_seed(0)
    attention = _SpatialAttention(64, 2, 16, pos_bias="rpb").eval()
    attention.local_na = True
    return attention, torch.randn(1, 64, 20, 18)


def test_a_triton_failure_switches_to_the_local_na_pytorch_form(monkeypatch):
    error = RuntimeError("kernel launch")
    error.__cause__ = _TritonError("ast.Num")
    attention, x = _local_na_failing_with(monkeypatch, error)
    with torch.no_grad():
        actual = attention.pre_gate_core(x)
        assert not dual_arch.LOCAL_NA_TRITON
        assert torch.equal(actual, attention.pre_gate_core(x))


def test_other_local_na_kernel_failures_are_raised(monkeypatch):
    attention, x = _local_na_failing_with(
        monkeypatch, torch.OutOfMemoryError("CUDA out of memory")
    )
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError):
        attention.pre_gate_core(x)
    assert dual_arch.LOCAL_NA_TRITON


def _routed_conv(channels=32, experts=4):
    source = torch.nn.Conv2d(channels, channels, 3, padding=1)
    base = dual_arch.EDBBReparamConv3x3(source)
    conv = dual_arch.PremixedDynamicGroupConv3x3(base, experts).double().eval()
    conv.experts.data.normal_(0, 0.05)
    for parameter in conv.router.parameters():
        parameter.data.normal_(0, 0.3)
    return conv


def _routing_reference(conv, x, reach=96):
    """FP64, one position at a time: the router reads the mean over the position's
    clamped neighborhood; the output is base + sum_k c_k(p) * conv_k."""
    channels, height, width = x.shape[1:]
    ky, kx = min(reach, height), min(reach, width)
    router = conv.router
    weight, bias = conv.base.equivalent_kernel(torch.float64)
    out = torch.nn.functional.conv2d(x, weight, bias, padding=1)
    experts = [
        torch.nn.functional.conv2d(x, e.reshape(channels, channels, 3, 3), padding=1)
        for e in conv.experts
    ]
    for y in range(height):
        for z in range(width):
            sy = min(max(y - ky // 2, 0), height - ky)
            sz = min(max(z - kx // 2, 0), width - kx)
            pooled = x[:, :, sy : sy + ky, sz : sz + kx].mean((2, 3))
            normalized = torch.nn.functional.layer_norm(pooled, (channels,))
            hidden = torch.nn.functional.gelu(router.fc1(normalized))
            c = router.fc2(hidden).softmax(-1) - 1 / router.num_experts
            for e, expert in enumerate(experts):
                out[:, :, y, z] += c[:, e, None] * expert[:, :, y, z]
    return out


# above 96 positions on one or both axes, batch 2, the folded convolution too
@pytest.mark.parametrize("shape", [(100, 7), (5, 130), (99, 98)])
def test_routing_na_matches_a_per_position_reference(shape):
    torch.manual_seed(0)
    conv = _routed_conv()
    conv.router.routing_na = True
    x = torch.randn(2, 32, *shape, dtype=torch.float64)
    with torch.no_grad():
        expected = _routing_reference(conv, x)
        torch.testing.assert_close(conv(x), expected, rtol=0, atol=1e-10)
        folded = conv.folded()
        assert folded.router.routing_na
        torch.testing.assert_close(folded(x), expected, rtol=0, atol=1e-10)


def test_routing_na_keeps_the_image_routing_up_to_96():
    torch.manual_seed(0)
    conv = _routed_conv().float()
    x = torch.randn(2, 32, 96, 80).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        expected = conv(x)
        conv.router.routing_na = True
        assert torch.equal(conv(x), expected)


def _gpu_routed_conv(channels, dtype):
    torch.manual_seed(0)
    conv = _routed_conv(channels).to("cuda", dtype)
    conv.router.routing_na = True
    return conv


# batch 2, non-square, above 96 positions on one or both axes, partial position and
# channel blocks; FP16 and the FP32 of the last group's convolution; both layouts
@needs_kernel
@pytest.mark.parametrize(
    ("batch", "channels", "shape", "dtype", "channels_last"),
    [
        (2, 128, (100, 7), torch.float16, False),
        (1, 64, (97, 130), torch.float16, True),
        (2, 48, (130, 99), torch.float16, False),
        (1, 128, (101, 120), torch.float32, False),
    ],
)
def test_routing_na_kernel_matches_the_pytorch_form(
    monkeypatch, batch, channels, shape, dtype, channels_last
):
    conv = _gpu_routed_conv(channels, dtype)
    x = torch.randn(batch, channels, *shape, device="cuda", dtype=dtype)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    calls = _counted(monkeypatch, "routed_sum")
    means = _counted(monkeypatch, "neighborhood_means")
    with torch.no_grad():
        actual = conv(x)
        coefficients = conv.router(x)
        assert len(calls) == 1 and len(means) == 2
        monkeypatch.setattr(dual_arch, "ROUTING_NA_TRITON", False)
        expected = conv(x)
        expected_coefficients = conv.router(x)
    assert len(calls) == 1 and len(means) == 2
    torch.testing.assert_close(coefficients, expected_coefficients, rtol=0, atol=1e-6)
    assert actual.dtype == dtype
    layout = torch.channels_last if channels_last else torch.contiguous_format
    assert actual.is_contiguous(memory_format=layout)
    atol = 2e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(actual, expected, rtol=0, atol=atol)


@needs_kernel
def test_routing_na_kernel_stays_off_up_to_96_and_outside_inference(monkeypatch):
    conv = _gpu_routed_conv(32, torch.float16)
    x = torch.randn(1, 32, 97, 40, device="cuda").half()
    calls = _counted(monkeypatch, "routed_sum")
    means = _counted(monkeypatch, "neighborhood_means")
    with torch.no_grad():
        conv(x[..., :96, :])
        conv.bfloat16()(x.bfloat16())
        conv.double()(x.double())
        conv.half()
        with torch.autocast("cuda", torch.float16):
            conv(x)
    conv(x)
    assert not calls and not means


def _routing_failing_with(monkeypatch, error):
    def launch(*_):
        raise error

    monkeypatch.setattr(dual_arch, "ROUTING_NA_TRITON", True)
    monkeypatch.setattr(dual_arch, "_na_kernel_usable", lambda *_: True)
    monkeypatch.setattr(na_local_triton, "routed_sum", launch)
    monkeypatch.setattr(na_local_triton, "neighborhood_means", launch)
    conv = _routed_conv().float()
    conv.router.routing_na = True
    return conv, torch.randn(1, 32, 100, 20)


def test_a_triton_failure_switches_to_the_routing_na_pytorch_form(monkeypatch):
    error = RuntimeError("kernel launch")
    error.__cause__ = _TritonError("ast.Num")
    conv, x = _routing_failing_with(monkeypatch, error)
    with torch.no_grad():
        actual = conv(x)
        assert not dual_arch.ROUTING_NA_TRITON
        assert torch.equal(actual, conv(x))


def test_other_routing_na_kernel_failures_are_raised(monkeypatch):
    conv, x = _routing_failing_with(
        monkeypatch, torch.OutOfMemoryError("CUDA out of memory")
    )
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError):
        conv(x)
    assert dual_arch.ROUTING_NA_TRITON


def _regional_reference(rca, q, k, v, reach=32):
    """FP64, one position at a time: channel_attention over the position's clamped
    neighborhood, applied to its own v."""
    batch, channels, height, width = q.shape
    heads, d = rca.heads, channels // rca.heads
    ky, kx = min(reach, height), min(reach, width)
    out = torch.empty_like(q)
    count = torch.tensor(float(ky * kx), dtype=torch.float64)
    for y in range(height):
        for z in range(width):
            sy = min(max(y - ky // 2, 0), height - ky)
            sz = min(max(z - kx // 2, 0), width - kx)

            def part(t, sy=sy, sz=sz):
                return t[:, :, sy : sy + ky, sz : sz + kx].reshape(batch, heads, d, -1)

            a = rca.channel_attention(part(q), part(k), count)
            own = v[:, :, y, z].reshape(batch, heads, d, 1)
            out[:, :, y, z] = (a @ own).reshape(batch, channels)
    return out


# above 16 positions on one or both axes, non-square, below and above the 32-wide
# neighborhood, batch 2; with a small cap every head runs in bands of a few rows
@pytest.mark.parametrize("bands", [False, True])
@pytest.mark.parametrize("shape", [(17, 16), (12, 40), (37, 20)])
def test_regional_na_matches_a_per_position_reference(monkeypatch, shape, bands):
    if bands:
        monkeypatch.setattr(dual_arch, "_NA_ELEMENT_CAP", 2 * 32 * 32 * 20 * 34)
    torch.manual_seed(0)
    rca = dual_arch.RegionalChannelAttention(64, 2).double().eval()
    rca.theta.data.normal_(0, 1)
    rca.regional_na = True
    q, k, v = torch.randn(3, 2, 64, *shape, dtype=torch.float64).unbind(0)
    with torch.no_grad():
        actual = rca.mix(q, k, v)
        expected = _regional_reference(rca, q, k, v)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


def test_regional_na_keeps_the_blend_up_to_16():
    torch.manual_seed(0)
    rca = dual_arch.RegionalChannelAttention(64, 2).eval()
    rca.theta.data.normal_(0, 1)
    for shape in ((16, 16), (16, 5), (9, 16)):
        q, k, v = torch.randn(3, 2, 64, *shape).unbind(0)
        with torch.no_grad():
            rca.regional_na = False
            expected = rca.mix(q, k, v)
            rca.regional_na = True
            assert torch.equal(rca.mix(q, k, v), expected)


# the condition of mix's shortcut: the four-class blend equals the per-position form
# up to 16 positions per axis, and not beyond
def test_the_blend_is_the_per_position_form_up_to_16():
    torch.manual_seed(0)
    rca = dual_arch.RegionalChannelAttention(64, 2).double().eval()
    rca.theta.data.normal_(0, 1)
    for shape, equal in (((16, 16), True), ((16, 9), True), ((17, 16), False)):
        q, k, v = torch.randn(3, 1, 64, *shape, dtype=torch.float64).unbind(0)
        with torch.no_grad():
            blend = rca.mix(q, k, v)
            error = (_regional_reference(rca, q, k, v) - blend).abs().max().item()
        assert (error < 1e-12) is equal, (shape, error)


def _na5_sites(model):
    return [
        m.routing_na if isinstance(m, dual_arch.DynamicRouter) else m.regional_na
        for m in model.modules()
        if isinstance(m, (dual_arch.DynamicRouter, dual_arch.RegionalChannelAttention))
    ]


def test_routing_and_regional_na_are_off_by_default_and_on_loading():
    on = _small(routing_na=True, regional_na=True)
    # two groups: two routers, four context layers
    assert _na5_sites(on) == [True] * 6
    for source in (dual_light(2), on):
        model = DUALArch().load(source.state_dict()).model
        assert not model.routing_na and not model.regional_na
        sites = _na5_sites(model)
        assert sites and not any(sites)
    assert not any(_na5_sites(dual_light(2)))
    with pytest.raises(ValueError, match="routing_na must be a bool"):
        _small(routing_na=1)


# switched off, the model is today's model: both switches on above both neighborhoods
# change the output, and switched back off it is bit-identical again
def test_routing_and_regional_na_off_is_todays_model():
    torch.manual_seed(0)
    model = _small(scale=2).eval()
    for name, parameter in model.named_parameters():
        if name.endswith("conv.experts"):
            parameter.data.normal_(0, 0.02)
    on = _small(scale=2, routing_na=True, regional_na=True).eval()
    on.load_state_dict(model.state_dict())
    image = torch.rand(2, 3, 101, 40)
    with torch.no_grad():
        expected = model(image)
        changed = on(image)
        for m in on.modules():
            if isinstance(m, dual_arch.DynamicRouter):
                m.routing_na = False
            if isinstance(m, dual_arch.RegionalChannelAttention):
                m.regional_na = False
        assert torch.equal(on(image), expected)
    assert (changed - expected).abs().max() > 1e-4


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# FP16 features, batch 2, non-square: the router and the channel attention calculate
# in FP32 and stay within FP16 rounding of the FP64 references
@needs_cuda
def test_routing_and_regional_na_in_fp16_on_the_gpu():
    torch.manual_seed(0)
    conv = _routed_conv(128)
    conv.router.routing_na = True
    x = torch.randn(2, 128, 120, 97, dtype=torch.float64)
    with torch.no_grad():
        expected = _routing_reference(conv, x)
        actual = conv.cuda().half()(x.cuda().half())
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual.double().cpu(), expected, rtol=0, atol=2e-2)

    rca = dual_arch.RegionalChannelAttention(128, 4).double().eval()
    rca.theta.data.normal_(0, 1)
    rca.regional_na = True
    q, k, v = torch.randn(3, 2, 128, 40, 37, dtype=torch.float64).unbind(0)
    with torch.no_grad():
        expected = _regional_reference(rca, q, k, v)
        actual = rca.cuda().half().mix(*(t.cuda().half() for t in (q, k, v)))
    # the Triton kernel writes FP16, the PyTorch form FP32 holding FP16 values
    assert actual.dtype == (
        torch.float16 if dual_arch.REGIONAL_NA_TRITON else torch.float32
    )
    torch.testing.assert_close(actual.double().cpu(), expected, rtol=0, atol=5e-3)


def _regional_operands(batch, channels, shape):
    """q, k and v as the depthwise convolution leaves them: chunks of one tensor."""
    torch.manual_seed(0)
    rca = dual_arch.RegionalChannelAttention(channels, channels // 32)
    rca.theta.data.normal_(0, 1)
    rca.regional_na = True
    qkv = torch.randn(batch, 3 * channels, *shape, device="cuda").half()
    return rca.cuda().eval().half(), qkv.chunk(3, dim=1)


# batch 2, non-square, above 16 positions on one or both axes, below and above the
# 32-wide neighborhood, partial row segments; chunk views and contiguous copies in
# both memory formats
@needs_kernel
@pytest.mark.parametrize(
    ("batch", "channels", "shape"),
    [
        (2, 128, (70, 41)),
        (1, 64, (17, 16)),
        (1, 64, (12, 40)),
        (2, 64, (37, 20)),
        (1, 32, (100, 7)),
    ],
)
def test_regional_na_kernel_matches_the_pytorch_form(
    monkeypatch, batch, channels, shape
):
    rca, (q, k, v) = _regional_operands(batch, channels, shape)
    calls = _counted(monkeypatch, "regional_na")
    with torch.no_grad():
        actual = rca.mix(q, k, v)
        copies = rca.mix(q.contiguous(), k.contiguous(), v.contiguous())
        channels_last = torch.cat((q, k, v), 1).contiguous(
            memory_format=torch.channels_last
        )
        assert channels_last.chunk(3, 1)[0].stride(1) == 1
        views = rca.mix(*channels_last.chunk(3, 1))
        assert len(calls) == 3
        monkeypatch.setattr(dual_arch, "REGIONAL_NA_TRITON", False)
        expected = rca.mix(q, k, v)
    assert len(calls) == 3
    assert torch.equal(actual, copies)
    # the channels-last build of the kernel may fuse other multiply-adds
    torch.testing.assert_close(views, actual, rtol=0, atol=1e-3)
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=2e-3)


@needs_kernel
def test_regional_na_kernel_stays_off_up_to_16_and_outside_inference(monkeypatch):
    rca, (q, k, v) = _regional_operands(1, 64, (20, 18))
    calls = _counted(monkeypatch, "regional_na")
    with torch.no_grad():
        rca.mix(*(t[..., :16, :16] for t in (q, k, v)))
        rca.float().mix(q.float(), k.float(), v.float())
        rca.bfloat16().mix(q.bfloat16(), k.bfloat16(), v.bfloat16())
        rca.half()
        with torch.autocast("cuda", torch.float16):
            rca.mix(q, k, v)
    rca.mix(q, k, v)
    assert not calls
    # heads of 64 channels keep the PyTorch form
    wide = dual_arch.RegionalChannelAttention(128, 2).cuda().eval().half()
    wide.regional_na = True
    with torch.no_grad():
        wide.mix(q.repeat(1, 2, 1, 1), k.repeat(1, 2, 1, 1), v.repeat(1, 2, 1, 1))
    assert not calls


# 2,056 x 1,024 positions in 65 row segments per column are 66,560 programs, past the
# 65,535 a second grid axis holds; the positions checked are the last programs' (the
# last segment of the last columns) and others across the map, against FP64.
@needs_kernel
def test_regional_na_kernel_covers_more_programs_than_a_grid_axis_holds():
    height, width = _GRID_SIDE, 1024
    rca, (q, k, v) = _regional_operands(1, 32, (height, width))
    reference = copy.deepcopy(rca).double().cpu()
    with torch.no_grad():
        out = rca.mix(q, k, v)
        for y, x in ((2055, 1023), (2054, 1022), (2049, 1000), (0, 0), (1031, 517)):
            sy, sx = min(max(y - 16, 0), height - 32), min(max(x - 16, 0), width - 32)
            span = [
                t[:, :, sy : sy + 32, sx : sx + 32].double().cpu() for t in (q, k, v)
            ]
            a = reference.channel_attention(
                span[0].reshape(1, 1, 32, 1024),
                span[1].reshape(1, 1, 32, 1024),
                torch.tensor(1024.0, dtype=torch.float64),
            )
            expected = (a[0, 0] @ span[2][0, :, y - sy, x - sx]).float()
            torch.testing.assert_close(
                out[0, :, y, x].float().cpu(), expected, rtol=0, atol=2e-3
            )


def _regional_failing_with(monkeypatch, error):
    def launch(*_):
        raise error

    monkeypatch.setattr(dual_arch, "REGIONAL_NA_TRITON", True)
    monkeypatch.setattr(dual_arch, "_na_kernel_usable", lambda *_: True)
    monkeypatch.setattr(na_local_triton, "regional_na", launch)
    torch.manual_seed(0)
    rca = dual_arch.RegionalChannelAttention(64, 2).eval()
    rca.regional_na = True
    return rca, torch.randn(3, 1, 64, 20, 18).unbind(0)


def test_a_triton_failure_switches_to_the_regional_na_pytorch_form(monkeypatch):
    error = RuntimeError("kernel launch")
    error.__cause__ = _TritonError("ast.Num")
    rca, (q, k, v) = _regional_failing_with(monkeypatch, error)
    with torch.no_grad():
        actual = rca.mix(q, k, v)
        assert not dual_arch.REGIONAL_NA_TRITON
        assert torch.equal(actual, rca.mix(q, k, v))


def test_other_regional_na_kernel_failures_are_raised(monkeypatch):
    rca, (q, k, v) = _regional_failing_with(
        monkeypatch, torch.OutOfMemoryError("CUDA out of memory")
    )
    with torch.no_grad(), pytest.raises(torch.OutOfMemoryError):
        rca.mix(q, k, v)
    assert dual_arch.REGIONAL_NA_TRITON
