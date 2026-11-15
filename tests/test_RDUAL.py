import importlib
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import pytest
import torch

from spandrel import MAIN_REGISTRY, ImageModelDescriptor
from spandrel.architectures.DUAL.__arch import na_local_triton
from spandrel.architectures.DUAL.__arch.dual_arch import dual_light
from spandrel.architectures.RDUAL import RDUAL, RDUALArch, run_clip, run_window
from spandrel.architectures.RDUAL.__arch import rdual_arch

from .util import assert_loads_correctly, skip_if_unchanged

skip_if_unchanged(__file__)


def _model(**kwargs) -> RDUAL:
    model = RDUAL(use_checkpoint=False, **kwargs)
    # Stands in for real flow weights, which a saved RDUAL generator always holds.
    model.get_buffer("flow_loaded").fill_(True)
    return model


def test_load():
    assert_loads_correctly(
        RDUALArch(),
        lambda: _model(scale=2),
        lambda: _model(scale=4),
        lambda: _model(scale=1),
        lambda: _model(scale=2, dynamic_experts=2),
    )


def test_registry_tells_rdual_from_dual():
    assert MAIN_REGISTRY.load(_model(scale=2).state_dict()).architecture.id == "RDUAL"
    assert MAIN_REGISTRY.load(dual_light(2).state_dict()).architecture.id == "DUAL"


def test_weights_without_flow_are_rejected():
    model = _model(scale=2)
    model.get_buffer("flow_loaded").fill_(False)
    with pytest.raises(ValueError, match="optical flow"):
        RDUALArch().load(model.state_dict())


def test_zero_temporal_stage_is_dual_light_on_every_frame():
    # A DUAL Light checkpoint in RDUAL: the temporal stage starts at zero output, so
    # every frame of a clip must come out as DUAL Light alone makes it.
    torch.manual_seed(0)
    dual = dual_light(2).eval()
    for parameter in dual.parameters():
        parameter.data.normal_(0, 0.02)
    rdual = _model(scale=2)
    missing, unexpected = rdual.load_state_dict(dual.state_dict(), strict=False)
    assert not unexpected
    assert all(key.startswith(("temporal.", "flow.", "flow_loaded")) for key in missing)
    rdual_descriptor = RDUALArch().load(rdual.state_dict())
    dual_descriptor = MAIN_REGISTRY.load(dual.state_dict())
    assert isinstance(dual_descriptor, ImageModelDescriptor)

    frames = torch.rand(1, 2, 3, 32, 40)
    with torch.inference_mode():
        clip = run_clip(rdual_descriptor.model, frames)
        alone = torch.stack([dual_descriptor(frames[:, t]) for t in range(2)], 1)
    assert clip.shape == (1, 2, 3, 64, 80)
    # The descriptor's call clamps to 0..1; run_clip returns the model's own values.
    torch.testing.assert_close(clip.clamp(0, 1), alone, atol=2e-5, rtol=0)


def test_one_image_is_a_clip_of_one_frame():
    descriptor = RDUALArch().load(_model(scale=2).state_dict())
    image = torch.rand(1, 3, 32, 40)
    with torch.inference_mode():
        out = descriptor(image)
        clip = run_clip(descriptor.model, image[:, None])[:, 0]
    assert out.shape == (1, 3, 64, 80)
    torch.testing.assert_close(out, clip.clamp(0, 1), atol=0, rtol=0)


def _temporal_model() -> RDUAL:
    # The temporal stage starts at zero output; random weights make every frame
    # read its neighbors and the carried state.
    torch.manual_seed(0)
    model = _model(scale=2).eval()
    with torch.no_grad():
        for parameter in model.temporal.parameters():
            parameter.normal_(0, 0.05)
    return model


def test_chained_windows_reaching_the_end_are_the_whole_clip():
    model = _temporal_model()
    frames = torch.rand(1, 7, 3, 32, 40)
    with torch.inference_mode():
        whole = model(frames)
        first, state = run_window(model, frames, 3)
        second, _ = run_window(model, frames[:, 3:], 4, (state, frames[:, 2]))
    assert first.shape == (1, 3, 3, 64, 80)
    assert second.shape == (1, 4, 3, 64, 80)
    torch.testing.assert_close(torch.cat((first, second), 1), whole)


def test_a_carried_state_changes_the_window_against_a_cold_start():
    model = _temporal_model()
    frames = torch.rand(1, 6, 3, 32, 40)
    with torch.inference_mode():
        _, state = run_window(model, frames[:, :3], 3)
        carried, _ = run_window(model, frames[:, 3:], 2, (state, frames[:, 2]))
        cold, _ = run_window(model, frames[:, 3:], 2)
    assert carried.shape == cold.shape == (1, 2, 3, 64, 80)
    assert (carried - cold).abs().amax() > 1e-3


def test_on_demand_flow_lookups_match_the_volume(monkeypatch):
    model = _temporal_model()
    frames = torch.rand(1, 3, 3, 48, 72)
    # Chunks smaller than one row of 1/8-scale queries split the lookups.
    monkeypatch.setattr(rdual_arch, "_LOOKUP_CHUNK_BYTES", 81 * 128 * 4 * 5)
    with torch.inference_mode():
        volume = model.guidance(frames)
        on_demand = model.guidance(frames, flow_volume=False)
    torch.testing.assert_close(on_demand, volume, rtol=1e-4, atol=1e-4)


def test_window_memory_modes(monkeypatch):
    model = _temporal_model()
    frames = torch.rand(1, 6, 3, 32, 40)
    volumes = []
    real = rdual_arch.NeuFlowV2.forward

    def forward(self, img0, img1, volume=True):
        volumes.append(volume)
        return real(self, img0, img1, volume)

    monkeypatch.setattr(rdual_arch.NeuFlowV2, "forward", forward)
    with torch.inference_mode():
        _, state = run_window(model, frames[:, :3], 2)
        past = (state, frames[:, 1])
        default, default_state = run_window(model, frames[:, 2:], 3, past)
        assert volumes and all(volumes)
        volumes.clear()
        recomputed, recomputed_state = run_window(
            model, frames[:, 2:], 3, past, hold_encodes=False
        )
        on_demand, _ = run_window(model, frames[:, 2:], 3, past, flow_volume=False)
    assert False in volumes
    # Encoding a frame again repeats the same operations on the same input.
    assert torch.equal(recomputed, default)
    assert torch.equal(recomputed_state, default_state)
    torch.testing.assert_close(on_demand, default, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("size", [(5, 9), (37, 50), (64, 42)])
def test_the_last_stage_in_bands_is_the_whole_frame_stage(monkeypatch, size):
    # Without gradients the deployed tail runs in bands; with them, on the whole
    # frame. Two rows of t2 per band, so bands meet inside the frame and at its edges.
    model = RDUAL(scale=2, use_checkpoint=False).eval()
    model.get_buffer("flow_loaded").fill_(True)
    model.prepare_for_export()
    monkeypatch.setattr(rdual_arch, "_TAIL_BAND_BYTES", 256 * size[1] * 4 * 2)
    frames = torch.rand(1, 2, 3, *size)
    guidance = torch.zeros(1, 1, 2, 2, (size[0] + 3) // 4 * 2, (size[1] + 3) // 4 * 2)
    with torch.no_grad():
        banded = model(frames, guidance)
    with torch.enable_grad():
        whole = model(frames, guidance)
    torch.testing.assert_close(banded, whole.detach())


def test_a_tile_that_covers_the_frame_is_the_untiled_window():
    model = _temporal_model()
    frames = torch.rand(1, 4, 3, 37, 50)
    with torch.inference_mode():
        untiled, state = run_window(model, frames, 3)
        covered, covered_state = run_window(model, frames, 3, tile=(40, 52))
    assert torch.equal(covered, untiled)
    assert torch.equal(covered_state, state)


def test_tile_starts_sit_on_the_grid_and_the_last_tile_ends_at_the_frame():
    for length, tile, overlap in ((837, 452, 64), (1488, 776, 64), (1080, 400, 32)):
        spans = rdual_arch.tile_axis(length, tile, overlap)
        padded = (length + 3) // 4 * 4
        assert len({size for _, size in spans}) == 1
        assert all(start % rdual_arch.TILE_GRID == 0 for start, _ in spans)
        assert spans[0][0] == 0 and spans[-1][0] + spans[-1][1] == padded
        assert spans[0][1] >= tile // 4 * 4
        assert all(
            a + size - b >= overlap for (a, size), (b, _) in zip(spans, spans[1:])
        )


@pytest.mark.parametrize("size", [(37, 50), (41, 29)])
def test_tiled_windows_carry_one_state_per_tile(size):
    # Odd sizes: the grid does not divide the padded frame evenly.
    model = _temporal_model()
    frames = torch.rand(1, 6, 3, *size)
    tile, overlap = (24, 20), 8
    grid = rdual_arch.tile_grid(size, tile, overlap)
    assert grid is not None
    rows, cols = grid
    assert len(rows) > 1 and len(cols) > 1
    assert len({span[1] for span in rows}) == len({span[1] for span in cols}) == 1
    with torch.inference_mode():
        first, state = run_window(model, frames[:, :3], 2, tile=tile, overlap=overlap)
        assert first.shape == (1, 2, 3, size[0] * 2, size[1] * 2)
        assert state.shape[0] == len(rows) * len(cols)
        past = (state, frames[:, 1])
        carried, _ = run_window(
            model, frames[:, 2:], 3, past, tile=tile, overlap=overlap
        )
        cold, _ = run_window(model, frames[:, 2:], 3, tile=tile, overlap=overlap)
        assert carried.shape == cold.shape == (1, 3, 3, size[0] * 2, size[1] * 2)
        assert (carried - cold).abs().amax() > 1e-3
        assert torch.isfinite(carried).all()
        # The tiles' states go together: not into an untiled window or other tiles.
        with pytest.raises(ValueError, match="tile"):
            run_window(model, frames[:, 2:], 3, past)
        with pytest.raises(ValueError, match="tile"):
            run_window(model, frames[:, 2:], 3, past, tile=(16, 16), overlap=4)
        _, untiled_state = run_window(model, frames[:, :3], 2)
        with pytest.raises(ValueError, match="tile"):
            run_window(
                model,
                frames[:, 2:],
                3,
                (untiled_state, frames[:, 1]),
                tile=tile,
                overlap=overlap,
            )


def test_the_attention_kernel_stays_off_on_rocm_builds(monkeypatch):
    cuda = torch.device("cuda")
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    monkeypatch.setattr(rdual_arch, "_HAS_DIRECT_ATTENTION", True)
    monkeypatch.setattr(torch.version, "hip", None)
    assert rdual_arch.na_kernel_usable(cuda, torch.bfloat16, 32)
    monkeypatch.setattr(torch.version, "hip", "7.0")
    assert not rdual_arch.na_kernel_usable(cuda, torch.bfloat16, 32)


@pytest.mark.parametrize("keep", [0, 4])
def test_a_window_keeps_one_to_all_of_its_frames(keep):
    with pytest.raises(ValueError, match="keep must be"):
        _temporal_model().infer_window(torch.rand(1, 3, 3, 32, 40), keep)


class _TritonError(Exception):
    pass


_TritonError.__module__ = "triton.compiler.errors"


def test_a_triton_failure_switches_to_the_pytorch_paths(monkeypatch):
    model = _model(scale=2)
    calls: list[str] = []
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)

    def forward(model: RDUAL, frames: torch.Tensor) -> torch.Tensor:
        calls.append(model.temporal.backend)
        if model.temporal.backend == "auto":
            raise RuntimeError("kernel launch") from _TritonError("ast.Num")
        return frames

    monkeypatch.setattr("spandrel.architectures.RDUAL._forward", forward)
    frames = torch.rand(1, 2, 3, 16, 16)
    assert run_clip(model, frames) is frames
    assert calls == ["auto", "reference"]
    assert model.temporal.backend == "reference"
    assert not rdual_arch.NA_TRITON


def test_other_failures_are_raised(monkeypatch):
    model = _model(scale=2)

    def forward(*_: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr("spandrel.architectures.RDUAL._forward", forward)
    with pytest.raises(torch.OutOfMemoryError):
        run_clip(model, torch.rand(1, 2, 3, 16, 16))
    assert model.temporal.backend == "auto"


def test_importing_spandrel_leaves_the_architecture_module_unloaded():
    # It imports Triton and Flex attention: only an RDUAL model pays for them.
    module = "spandrel.architectures.RDUAL.__arch.rdual_arch"
    code = "\n".join(
        [
            "import sys, spandrel",
            "from spandrel.architectures import RDUAL",
            f"assert {module!r} not in sys.modules",
            "assert RDUAL.RDUAL.__name__ == 'RDUAL'",
            f"assert {module!r} in sys.modules",
        ]
    )
    spandrel = Path(__file__).parent.parent / "libs" / "spandrel"
    env = {**os.environ, "PYTHONPATH": str(spandrel)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=240)


def test_the_export_names_are_the_architecture_modules_own():
    package = importlib.import_module("spandrel.architectures.RDUAL")

    pairs = {
        "HAS_DIRECT_ATTENTION": "_HAS_DIRECT_ATTENTION",
        "DIRECT_ATTENTION_ERROR": "_DIRECT_ATTENTION_ERROR",
        "SpatialAttention": "_SpatialAttention",
        "GROUP_INDICES": "_INDICES",
        **{
            f"NF_{kind}_{level}": f"_NF_{kind}_{level}"
            for kind in ("CONTEXT_DIM", "FEATURE_DIM", "ITER_CONTEXT_DIM", "ITERS")
            for level in ("S16", "S8")
        },
    }
    if rdual_arch.HAS_DIRECT_ATTENTION:
        pairs["TEMPORAL_FORWARD_KERNEL"] = "_temporal_forward_kernel"
        pairs["KV_WARP_FORWARD_KERNEL"] = "_kv_warp_forward_kernel"
    else:
        assert package.TEMPORAL_FORWARD_KERNEL is None
        assert package.KV_WARP_FORWARD_KERNEL is None
    for public, private in pairs.items():
        assert getattr(package, public) is getattr(rdual_arch, private), public
    assert package.NeuFlowV2 is rdual_arch.NeuFlowV2
    neuflow = rdual_arch.NeuFlowV2
    assert neuflow.prepare_grids is vars(neuflow)["_prepare_grids"]


@pytest.mark.skipif(
    not (
        hasattr(rdual_arch, "direct_attention")
        and torch.cuda.is_available()
        and torch.version.hip is None
    ),
    reason="the direct attention runs on NVIDIA GPUs with Triton",
)
def test_direct_attention_takes_more_windows_than_a_grid_axis_holds():
    # 16,500 windows of 4 heads: 66,000 programs on the grid's first axis; the same
    # windows in two calls under the limit give the same values
    torch.manual_seed(0)
    window, heads, half = 4, 4, 8_250
    q, k, v = (
        torch.randn(2 * half, heads, window * window, 16, device="cuda").half()
        for _ in range(3)
    )
    table = torch.randn((2 * window - 1) ** 2, heads, device="cuda").half()
    with torch.no_grad():
        whole = rdual_arch.direct_attention(q, k, v, table, window)
        parts = [
            rdual_arch.direct_attention(
                q[i : i + half], k[i : i + half], v[i : i + half], table, window
            )
            for i in (0, half)
        ]
    assert torch.equal(whole, torch.cat(parts))


def test_every_rdual_loads_with_local_and_retrieval_na():
    model = RDUALArch().load(_model(scale=2).state_dict()).model
    assert model.local_na and model.retrieval_na
    layers = [
        a.local_na
        for a in model.modules()
        if isinstance(a, rdual_arch.SpatialAttention) and not a.context
    ]
    assert layers and all(layers)


def _starts(length: int, keys: int, size: int, stride: int) -> torch.Tensor:
    return (torch.arange(length) // stride - size // 2).clamp(0, keys - size)


# above the window on one or both axes, partial 8x8 query tiles, batch 2
@pytest.mark.parametrize("shape", [(17, 30), (40, 9)])
def test_local_na_matches_dense_reference(shape):
    """Dense FP64 attention in which every query sees only its own clamped 16x16
    neighborhood, with the dense RPB of the query-minus-key offset."""
    torch.manual_seed(0)
    attention = rdual_arch.SpatialAttention(64, 2, 16, shifted=True, context=False)
    attention = attention.double().eval()
    assert attention.relative_bias is not None
    attention.relative_bias.data.normal_(0, 0.5)
    attention.local_na = True
    x = torch.randn(2, 64, *shape, dtype=torch.float64)
    height, width = shape
    kh, kw = min(height, 16), min(width, 16)
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    y, z = yy.flatten(), xx.flatten()
    sy, sz = _starts(height, height, kh, 1)[y], _starts(width, width, kw, 1)[z]
    inside = (y >= sy[:, None]) & (y < sy[:, None] + kh)
    inside &= (z >= sz[:, None]) & (z < sz[:, None] + kw)
    dy = (y[:, None] - y + 15).clamp(0, 30)
    dz = (z[:, None] - z + 15).clamp(0, 30)
    bias = attention.relative_bias[dy * 31 + dz].permute(2, 0, 1)
    with torch.no_grad():
        actual = attention.pre_gate_core(x)
        qkv = attention.qkv(x.flatten(2).transpose(1, 2)).reshape(2, -1, 3, 2, 32)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        logits = query @ key.transpose(-1, -2) * 32**-0.5 + bias
        weights = logits.masked_fill(~inside, -math.inf).softmax(-1)
        expected = (weights @ value).transpose(-1, -2).reshape(x.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("coarse", [(49, 30), (30, 52)])
def test_retrieval_na_matches_dense_reference(coarse):
    """Fine query (y, x) attends the 48x48 coarse keys around (y // 2, x // 2),
    clamped inside the coarse grid."""
    torch.manual_seed(0)
    ch, cw = coarse
    q = torch.randn(2, 64, 2 * ch, 2 * cw, dtype=torch.float64)
    k, v = (torch.randn(2, 64, ch, cw, dtype=torch.float64) for _ in range(2))
    log_tau = torch.randn(2, dtype=torch.float64) * 0.3
    packed = rdual_arch.pack_qk(q, k, na=True)
    assert isinstance(packed[0], rdual_arch.NAGeometry)
    actual = rdual_arch.attend_packed(packed, v, log_tau)
    kh, kw = min(48, ch), min(48, cw)
    yy, xx = torch.meshgrid(torch.arange(2 * ch), torch.arange(2 * cw), indexing="ij")
    ky, kx = torch.meshgrid(torch.arange(ch), torch.arange(cw), indexing="ij")
    ky, kx = ky.flatten(), kx.flatten()
    sy = _starts(2 * ch, ch, kh, 2)[yy.flatten()]
    sz = _starts(2 * cw, cw, kw, 2)[xx.flatten()]
    inside = (ky >= sy[:, None]) & (ky < sy[:, None] + kh)
    inside &= (kx >= sz[:, None]) & (kx < sz[:, None] + kw)
    tau = log_tau.exp().clamp(0.1, 10)[None, :, None, None]
    query, key, value = (t.reshape(2, 2, 32, -1).transpose(-1, -2) for t in (q, k, v))
    scale = math.log(kh * kw) / (math.log(1024) * math.sqrt(32))
    logits = (query * tau @ key.transpose(-1, -2) * scale).masked_fill(
        ~inside, -math.inf
    )
    expected = (logits.softmax(-1) @ value).transpose(-1, -2).reshape(q.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


def test_retrieval_na_keeps_the_blocks_up_to_48():
    q = torch.randn(1, 64, 96, 90)
    k = torch.randn(1, 64, 48, 45)
    packed = rdual_arch.pack_qk(q, k, na=True)
    assert not isinstance(packed[0], rdual_arch.NAGeometry)
    v = torch.randn(1, 64, 48, 45)
    log_tau = torch.zeros(2)
    assert torch.equal(
        rdual_arch.attend_packed(packed, v, log_tau),
        rdual_arch.attend_packed(rdual_arch.pack_qk(q, k), v, log_tau),
    )


needs_kernel = pytest.mark.skipif(
    not (
        rdual_arch.HAS_DIRECT_ATTENTION
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


# regional_na is off by default: the regional kernel never launches, the routing
# kernels do
@needs_kernel
def test_by_default_rdual_runs_no_regional_kernel(monkeypatch):
    regional = _counted(monkeypatch, "regional_na")
    routed = _counted(monkeypatch, "routed_sum")
    model = _model(scale=2).cuda().eval()
    assert not any(m.regional_na for m in _regionals(model))
    with torch.inference_mode():
        run_clip(model, torch.rand(1, 2, 3, 208, 200, device="cuda"))
        run_clip(
            model.bfloat16(), torch.rand(1, 1, 3, 208, 200, device="cuda").bfloat16()
        )
    assert not regional and routed


def _local_attention(channels):
    attention = rdual_arch.SpatialAttention(
        channels, channels // 32, 16, shifted=False, context=False
    )
    assert attention.relative_bias is not None
    attention.relative_bias.data.normal_(0, 0.5)
    attention.local_na = True
    return attention.cuda().eval().bfloat16()


# RDUAL's BF16: batch 2, non-square, partial 8x8 query tiles, above the window on
# one or both axes
@needs_kernel
@pytest.mark.parametrize(
    ("batch", "channels", "shape"),
    [(2, 128, (40, 37)), (1, 128, (17, 200)), (1, 64, (96, 9))],
)
def test_local_na_kernel_matches_the_pytorch_form(monkeypatch, batch, channels, shape):
    torch.manual_seed(0)
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    attention = _local_attention(channels)
    x = torch.randn(batch, channels, *shape, device="cuda").bfloat16()
    calls = _counted(monkeypatch, "local_na")
    with torch.no_grad():
        actual = attention.pre_gate_core(x)
        assert len(calls) == 1
        monkeypatch.setattr(rdual_arch, "NA_TRITON", False)
        expected = attention.pre_gate_core(x)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)


@needs_kernel
def test_local_na_kernel_stays_off_up_to_the_window_and_outside_inference(
    monkeypatch,
):
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    attention = _local_attention(64)
    calls = _counted(monkeypatch, "local_na")
    with torch.no_grad():
        attention.pre_gate_core(torch.randn(1, 64, 16, 12, device="cuda").bfloat16())
        attention.float().pre_gate_core(torch.randn(1, 64, 17, 17, device="cuda"))
    attention.bfloat16().pre_gate_core(
        torch.randn(1, 64, 17, 17, device="cuda").bfloat16()
    )
    assert not calls


# torch.jit.trace records the PyTorch form: a kernel launch would leave the traced
# graph reading an output buffer it never fills
@needs_kernel
def test_traced_calls_take_the_pytorch_form(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    # the traced function closes over the weights: frozen, they become constants
    attention = _local_attention(64).requires_grad_(False)
    x = torch.randn(1, 64, 20, 18, device="cuda").bfloat16()
    q = torch.randn(1, 128, 100, 98, device="cuda").bfloat16()
    k, v = (torch.randn(1, 128, 50, 49, device="cuda").bfloat16() for _ in "kv")
    log_tau = torch.randn(4, device="cuda") * 0.3

    def local_na(x):
        return attention.pre_gate_core(x)

    def retrieval(q, k, v):
        return rdual_arch.attend_packed(rdual_arch.pack_qk(q, k, na=True), v, log_tau)

    local, retrieved = (
        _counted(monkeypatch, "local_na"),
        _counted(monkeypatch, "retrieval_na"),
    )
    with torch.no_grad():
        traced_local = torch.jit.trace(local_na, x, check_trace=False)
        traced_retrieval = torch.jit.trace(retrieval, (q, k, v), check_trace=False)
        assert isinstance(traced_local, torch.jit.ScriptFunction)
        assert isinstance(traced_retrieval, torch.jit.ScriptFunction)
        assert not local and not retrieved
        monkeypatch.setattr(rdual_arch, "NA_TRITON", False)
        assert torch.equal(traced_local(x), attention.pre_gate_core(x))
        assert torch.equal(traced_retrieval(q, k, v), retrieval(q, k, v))


# coarse grids above 48 on one or both axes, odd fine sizes, batch 2; 48 x 45 keeps
# the blocks
@needs_kernel
@pytest.mark.parametrize(
    ("batch", "fine", "coarse", "kernel"),
    [
        (2, (100, 98), (50, 49), True),
        (1, (30, 104), (15, 52), True),
        (1, (97, 40), (49, 20), True),
        (1, (96, 90), (48, 45), False),
    ],
)
def test_retrieval_na_kernel_matches_the_pytorch_form(
    monkeypatch, batch, fine, coarse, kernel
):
    torch.manual_seed(0)
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    q = torch.randn(batch, 128, *fine, device="cuda").bfloat16()
    k, v = (torch.randn(batch, 128, *coarse, device="cuda").bfloat16() for _ in "kv")
    log_tau = torch.randn(4, device="cuda") * 0.3
    calls = _counted(monkeypatch, "retrieval_na")
    with torch.no_grad():
        actual = rdual_arch.attend_packed(rdual_arch.pack_qk(q, k, na=True), v, log_tau)
        assert len(calls) == int(kernel)
        monkeypatch.setattr(rdual_arch, "NA_TRITON", False)
        expected = rdual_arch.attend_packed(
            rdual_arch.pack_qk(q, k, na=True), v, log_tau
        )
    assert len(calls) == int(kernel)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)


def _routers(model: RDUAL):
    return [m for m in model.modules() if isinstance(m, rdual_arch.DynamicRouter)]


def test_routing_na_is_on_by_default_and_on_loading():
    off = _routers(_model(scale=2, routing_na=False))
    assert len(off) == 4 and not any(router.routing_na for router in off)
    model = RDUALArch().load(_model(scale=2).state_dict()).model
    assert model.routing_na
    routers = _routers(model)
    assert len(routers) == 4 and all(router.routing_na for router in routers)
    with pytest.raises(ValueError, match="routing_na must be a bool"):
        _model(scale=2, routing_na=1)


def _routed_conv(channels=32, experts=4):
    source = torch.nn.Conv2d(channels, channels, 3, padding=1)
    base = rdual_arch.EDBBReparamConv3x3(source)
    conv = rdual_arch.PremixedDynamicGroupConv3x3(base, experts).double().eval()
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
def test_routing_na_matches_a_per_position_reference(monkeypatch, shape):
    # the per-position form outside its kernels' devices, as the gate would allow it
    monkeypatch.setattr(rdual_arch, "_na_forms_available", lambda *_: True)
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


def test_routing_na_keeps_the_image_routing_up_to_96(monkeypatch):
    monkeypatch.setattr(rdual_arch, "_na_forms_available", lambda *_: True)
    torch.manual_seed(0)
    conv = _routed_conv().float()
    x = torch.randn(2, 32, 96, 80).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        expected = conv(x)
        conv.router.routing_na = True
        assert torch.equal(conv(x), expected)


def _gpu_routed_conv(channels):
    torch.manual_seed(0)
    conv = _routed_conv(channels).float().cuda().folded()
    conv.router.routing_na = True
    return conv


# RDUAL's precision (FP32 weights under BF16 autocast, as run_clip and run_window run
# the trunk) and BF16 weights; batch 2, non-square, above 96 positions on one or both
# axes (one at most 96 and the other above), partial position and channel blocks, both
# layouts
@needs_kernel
@pytest.mark.parametrize(
    ("batch", "channels", "shape", "bfloat16", "channels_last"),
    [
        (2, 128, (100, 7), False, False),
        (1, 128, (97, 130), False, True),
        (2, 48, (130, 99), True, False),
        (1, 64, (60, 200), False, False),
    ],
)
def test_routing_na_kernel_matches_the_pytorch_form(
    monkeypatch, batch, channels, shape, bfloat16, channels_last
):
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    conv = _gpu_routed_conv(channels)
    x = torch.randn(batch, channels, *shape, device="cuda")
    if bfloat16:
        conv, x = conv.bfloat16(), x.bfloat16()
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    calls = _counted(monkeypatch, "routed_sum")
    means = _counted(monkeypatch, "neighborhood_means")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=not bfloat16):
        actual = conv(x)
        coefficients = conv.router(x)
        assert len(calls) == 1 and len(means) == 2
        monkeypatch.setattr(rdual_arch, "NA_TRITON", False)
        expected = conv(x)
        expected_coefficients = conv.router(x)
    assert len(calls) == 1 and len(means) == 2
    torch.testing.assert_close(coefficients, expected_coefficients, rtol=0, atol=1e-6)
    assert actual.dtype == torch.bfloat16
    layout = torch.channels_last if channels_last else torch.contiguous_format
    assert actual.is_contiguous(memory_format=layout)
    # one BF16 step apart at most (the sums' order differs)
    torch.testing.assert_close(actual, expected, rtol=2**-7, atol=1e-3)


# up to 96, training, gradients, FP32, FP16 (autocast or weights), FP64: image routing
@needs_kernel
def test_routing_na_stays_off_up_to_96_and_outside_bf16_inference(monkeypatch):
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    conv = _gpu_routed_conv(32)
    x = torch.randn(1, 32, 97, 40, device="cuda")
    calls = _counted(monkeypatch, "routed_sum")
    means = _counted(monkeypatch, "neighborhood_means")
    bf16 = partial(torch.autocast, "cuda", torch.bfloat16)
    fp16 = partial(torch.autocast, "cuda", torch.float16)

    def image_routing(conv, x, context):
        # the form is in the coefficients' shape; the output must still run
        with context():
            routed = conv.router(x)
            conv(x)
        return routed.shape[-2:] == (1, 1)

    with torch.no_grad():
        assert image_routing(conv, x[..., :96, :], bf16)
        assert image_routing(conv.train(), x, bf16)
        conv.eval()
        assert image_routing(conv, x, nullcontext)
        assert image_routing(conv, x, fp16)
        assert image_routing(conv.half(), x.half(), nullcontext)
        # autocast leaves FP64 alone: an FP64 model under BF16 autocast runs in FP64
        assert image_routing(conv.double(), x.double(), bf16)
        conv.float()
        assert not image_routing(conv, x, bf16)
        calls.clear()
        means.clear()
    assert image_routing(conv, x, bf16)
    assert not calls and not means


@needs_kernel
def test_routing_na_blends_in_pytorch_where_routed_sum_does_not_fit(monkeypatch):
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    conv = _gpu_routed_conv(64)
    x = torch.randn(2, 64, 110, 70, device="cuda")
    calls = _counted(monkeypatch, "routed_sum")
    means = _counted(monkeypatch, "neighborhood_means")
    monkeypatch.setattr(na_local_triton, "routed_sum_fits", lambda *_: False)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        actual = conv(x)
        assert not calls and len(means) == 1
        monkeypatch.setattr(rdual_arch, "NA_TRITON", False)
        expected = conv(x)
    assert actual.dtype == torch.bfloat16
    # the same blend; only the means' kernel differs (FP32 rounding of the router)
    torch.testing.assert_close(actual, expected, rtol=2**-7, atol=1e-3)


def _routed_model():
    torch.manual_seed(0)
    model = _model(scale=2).cuda().eval()
    for conv in model.modules():
        if isinstance(conv, rdual_arch.PremixedDynamicGroupConv3x3):
            conv.experts.data.normal_(0, 0.05)
            for parameter in conv.router.parameters():
                parameter.data.normal_(0, 0.3)
    return model


def _per_position_forward(self, x):
    """FP32, vectorized: means over the clamped 96x96 neighborhoods by box sums and
    replicated borders, the router per position, base + sum_k c_k * conv_k; returns
    the output and the coefficients B,H,W,K."""
    channels, height, width = x.shape[1:]
    ky, kx = min(96, height), min(96, width)
    with torch.autocast("cuda", enabled=False):
        x = x.float()
        weight, bias = self.base.equivalent_kernel(torch.float32)
        means = torch.nn.functional.avg_pool2d(x, (ky, kx), 1)
        means = torch.nn.functional.pad(
            means, (kx // 2, kx - 1 - kx // 2, ky // 2, ky - 1 - ky // 2), "replicate"
        ).permute(0, 2, 3, 1)
        router = self.router
        hidden = torch.nn.functional.gelu(
            router.fc1(torch.nn.functional.layer_norm(means, (channels,)))
        )
        c = router.fc2(hidden).softmax(-1) - 1 / router.num_experts
        out = torch.nn.functional.conv2d(x, weight, bias, padding=1)
        for e, expert in enumerate(self.experts):
            kernel = expert.reshape(channels, channels, 3, 3)
            out += c[..., e][:, None] * torch.nn.functional.conv2d(x, kernel, padding=1)
    return out, c


# the app's call in BF16 on a frame whose trunk is 300 x 32 (one axis above 96): every
# routed convolution's output against the reference on the same input
@needs_kernel
def test_run_window_routes_per_position_like_a_reference(monkeypatch):
    model = _routed_model()
    # content that changes along the tall axis, so that neighborhoods differ
    ramp = torch.linspace(0, 1, 600, device="cuda")[:, None]
    frames = (ramp * torch.rand(1, 2, 3, 1, 64, device="cuda")).expand(1, 2, 3, 600, 64)
    frames = frames + 0.1 * torch.rand(1, 2, 3, 600, 64, device="cuda")
    routed = _counted(monkeypatch, "routed_sum")
    seen = []
    for conv in model.modules():
        if isinstance(conv, rdual_arch.PremixedDynamicGroupConv3x3):

            def record(x, conv=conv, forward=conv.forward):
                out = forward(x)
                seen.append((conv, x, out))
                return out

            monkeypatch.setattr(conv, "forward", record)
    with torch.inference_mode():
        run_window(model, frames, 2)
        assert len(routed) == len(seen) >= 4
        spread = 0.0
        for conv, x, out in seen:
            expected, coefficients = _per_position_forward(conv, x)
            # the router under the model run's autocast, as the site called it
            with torch.autocast("cuda", torch.bfloat16):
                routes = conv.router(x)
            assert routes.shape[-2:] == x.shape[-2:]
            # the routes match the reference's to FP32 rounding
            spread = max(spread, coefficients.std((1, 2)).max().item())
            torch.testing.assert_close(
                routes.permute(0, 2, 3, 1), coefficients, rtol=0, atol=1e-5
            )
            # the output to the BF16 convolution's rounding
            assert out.dtype == torch.bfloat16
            error = (out.float() - expected).abs().mean().item()
            assert error < 2**-7 * expected.abs().mean().item()
    # the routes vary over the map
    assert spread > 1e-2


# the app's window in BF16: a failing routing kernel makes run_window fall back to the
# PyTorch paths (NA_TRITON off), which route per position too
@needs_kernel
@pytest.mark.parametrize("failing", ["routed_sum", "neighborhood_means"])
def test_a_routing_kernel_failure_falls_back_to_the_pytorch_form(monkeypatch, failing):
    model = _routed_model()
    frames = torch.rand(1, 2, 3, 208, 200, device="cuda")
    model.temporal.backend = "reference"
    monkeypatch.setattr(rdual_arch, "NA_TRITON", False)
    with torch.inference_mode():
        expected, _ = run_window(model, frames, 2)
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)

    def launch(*_):
        error = RuntimeError("kernel launch")
        error.__cause__ = _TritonError("ast.Num")
        raise error

    monkeypatch.setattr(na_local_triton, failing, launch)
    with torch.inference_mode():
        actual, _ = run_window(model, frames, 2)
    assert not rdual_arch.NA_TRITON
    assert torch.equal(actual, expected)
    with torch.inference_mode():
        model.routing_na = False
        for router in _routers(model):
            router.routing_na = False
        image, _ = run_window(model, frames, 2)
    assert not torch.equal(image, expected)


def _regionals(model: RDUAL):
    return [
        m for m in model.modules() if isinstance(m, rdual_arch.RegionalChannelAttention)
    ]


def test_regional_na_is_off_by_default_and_on_loading():
    default = _regionals(_model(scale=2))
    assert default and not any(m.regional_na for m in default)
    model = RDUALArch().load(_model(scale=2).state_dict()).model
    assert not model.regional_na
    assert not any(m.regional_na for m in _regionals(model))
    on = _model(scale=2, regional_na=True)
    assert on.regional_na and all(m.regional_na for m in _regionals(on))
    with pytest.raises(ValueError, match="regional_na must be a bool"):
        _model(scale=2, regional_na=1)


def _regional_attention():
    torch.manual_seed(0)
    rca = rdual_arch.RegionalChannelAttention(64, 2).eval()
    rca.theta.data.normal_(0, 1)
    rca.regional_na = True
    return rca


def _regional_reference(rca, q, k, v, reach=32):
    """FP64, one position at a time: channel_attention over the clamped neighborhood
    of each position, applied to its own v."""
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
    # the per-position form outside its kernel's devices, as the gate would allow it
    monkeypatch.setattr(rdual_arch, "_na_forms_available", lambda *_: True)
    if bands:
        monkeypatch.setattr(rdual_arch, "_REGIONAL_ELEMENT_CAP", 2 * 32 * 32 * 20 * 34)
    rca = _regional_attention().double()
    q, k, v = torch.randn(3, 2, 64, *shape, dtype=torch.float64).unbind(0)
    with torch.no_grad():
        actual = rca.mix(q, k, v)
        expected = _regional_reference(rca, q, k, v)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


def test_regional_na_keeps_the_blend_up_to_16(monkeypatch):
    monkeypatch.setattr(rdual_arch, "_na_forms_available", lambda *_: True)
    rca = _regional_attention()
    for shape in ((16, 16), (16, 5), (9, 16)):
        q, k, v = torch.randn(3, 2, 64, *shape).unbind(0)
        with torch.no_grad():
            rca.regional_na = False
            expected = rca.mix(q, k, v)
            rca.regional_na = True
            assert torch.equal(rca.mix(q, k, v), expected)


def _gpu_regional(channels=128):
    torch.manual_seed(0)
    rca = rdual_arch.RegionalChannelAttention(channels, channels // 32).cuda().eval()
    rca.theta.data.normal_(0, 1)
    rca.regional_na = True
    return rca


# RDUAL's precision (FP32 weights under BF16 autocast: q, k and v are BF16); batch 2,
# non-square, above 16 on one or both axes, partial 32-row segments, both layouts
@needs_kernel
@pytest.mark.parametrize(
    ("batch", "shape", "channels_last"),
    [(2, (40, 37), False), (1, (17, 200), True), (1, (100, 9), False)],
)
def test_regional_na_kernel_matches_the_pytorch_form(
    monkeypatch, batch, shape, channels_last
):
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    rca = _gpu_regional()
    x = torch.randn(batch, 128, *shape, device="cuda")
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    calls = _counted(monkeypatch, "regional_na")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        actual = rca(x)
        assert len(calls) == 1
        monkeypatch.setattr(rdual_arch, "NA_TRITON", False)
        expected = rca(x)
    assert len(calls) == 1
    # the kernel rounds the blend to BF16, as the projection's autocast does
    torch.testing.assert_close(actual, expected, rtol=2**-7, atol=2e-3)


# up to 16, training, gradients, FP32, FP16 (autocast or weights), FP64 under BF16
# autocast: the blend, bit for bit, and no launch
@needs_kernel
def test_regional_na_stays_off_up_to_16_and_outside_bf16_inference(monkeypatch):
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)
    rca = _gpu_regional()
    x = torch.randn(1, 128, 40, 30, device="cuda")
    calls = _counted(monkeypatch, "regional_na")
    bf16 = partial(torch.autocast, "cuda", torch.bfloat16)
    fp16 = partial(torch.autocast, "cuda", torch.float16)

    def blend(rca, x, context):
        with context():
            rca.regional_na = True
            actual = rca(x)
            rca.regional_na = False
            expected = rca(x)
        rca.regional_na = True
        return torch.equal(actual, expected)

    with torch.no_grad():
        assert blend(rca, x[..., :16, :16], bf16)
        assert blend(rca.train(), x, bf16)
        rca.eval()
        assert blend(rca, x, nullcontext)
        assert blend(rca, x, fp16)
        assert blend(rca.half(), x.half(), nullcontext)
        assert blend(rca.double(), x.double(), bf16)
        rca.float()
    assert blend(rca, x, bf16)
    assert not calls


# both switches on: one call takes both per-position forms (BF16 run_window) or neither
# (FP32, bit for bit the switches off)
@needs_kernel
def test_routing_and_regional_na_take_their_forms_together(monkeypatch):
    model = _routed_model()
    for m in _regionals(model):
        m.regional_na = True
    frames = torch.rand(1, 2, 3, 208, 200, device="cuda")
    routed = _counted(monkeypatch, "routed_sum")
    regional = _counted(monkeypatch, "regional_na")
    with torch.inference_mode():
        run_window(model, frames, 2)
        assert routed and regional
        routed.clear()
        regional.clear()
        both, _ = model.infer_window(frames, 2)
        assert not routed and not regional
        for router in _routers(model):
            router.routing_na = False
        for m in _regionals(model):
            m.regional_na = False
        off, _ = model.infer_window(frames, 2)
    assert torch.equal(both, off)


# a failing regional kernel makes run_window fall back to the PyTorch paths (NA_TRITON
# off), which take both per-position forms too
@needs_kernel
def test_a_regional_kernel_failure_falls_back_to_the_pytorch_form(monkeypatch):
    model = _routed_model()
    for m in _regionals(model):
        m.regional_na = True
    frames = torch.rand(1, 2, 3, 208, 200, device="cuda")
    model.temporal.backend = "reference"
    monkeypatch.setattr(rdual_arch, "NA_TRITON", False)
    with torch.inference_mode():
        expected, _ = run_window(model, frames, 2)
    monkeypatch.setattr(rdual_arch, "NA_TRITON", True)

    def launch(*_):
        error = RuntimeError("kernel launch")
        error.__cause__ = _TritonError("ast.Num")
        raise error

    monkeypatch.setattr(na_local_triton, "regional_na", launch)
    with torch.inference_mode():
        actual, _ = run_window(model, frames, 2)
    assert not rdual_arch.NA_TRITON
    assert torch.equal(actual, expected)
    # the control: with regional_na off the same run takes the blend
    with torch.inference_mode():
        for m in _regionals(model):
            m.regional_na = False
        blend, _ = run_window(model, frames, 2)
    assert not torch.equal(blend, expected)


# bands of 3 rows over 10 (a partial last band), batch 2, both layouts: the banded
# blend is the whole tensor's expression bit for bit
@pytest.mark.parametrize("channels_last", [False, True])
def test_plain_routed_sum_in_bands_is_the_whole_expression(monkeypatch, channels_last):
    torch.manual_seed(0)
    count, channels, height, width = 3, 16, 10, 12
    layout = torch.channels_last if channels_last else torch.contiguous_format
    parts = torch.randn(2, (count + 1) * channels, height, width).bfloat16()
    parts = parts.contiguous(memory_format=layout)
    coefficients = torch.randn(2, height, width, count + 1).permute(0, 3, 1, 2)
    coefficients = coefficients[:, :count]
    split = parts.float().unflatten(1, (count + 1, channels))
    expected = (split[:, 0] + (coefficients[:, :, None] * split[:, 1:]).sum(1)).to(
        parts.dtype
    )
    monkeypatch.setattr(na_local_triton, "PLAIN_ROUTED_POSITIONS", 3 * width)
    actual = na_local_triton.plain_routed_sum(
        parts, coefficients, torch.float32, layout
    )
    assert torch.equal(actual, expected)
    assert actual.is_contiguous(memory_format=layout)


# channel blocks of 3 over 7 (a partial last block), batch 2, both layouts, FP32 and
# BF16 input: the blocked means are the unblocked ones bit for bit, and the window means
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("channels_last", [False, True])
def test_plain_neighborhood_means_in_channel_blocks(monkeypatch, dtype, channels_last):
    torch.manual_seed(0)
    x = (torch.randn(2, 7, 30, 20) * 10).to(dtype)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    whole = na_local_triton.plain_neighborhood_means(x, 9, 6, torch.float32)
    monkeypatch.setattr(na_local_triton, "PLAIN_MEANS_ELEMENTS", 3 * 2 * 30 * 20)
    blocked = na_local_triton.plain_neighborhood_means(x, 9, 6, torch.float32)
    assert torch.equal(blocked, whole)
    windows = torch.nn.functional.avg_pool2d(x.double(), (9, 6), 1).permute(0, 2, 3, 1)
    torch.testing.assert_close(blocked.double(), windows, rtol=2**-23, atol=1e-6)


# the form follows the site's weights and the call's context, never the features'
# dtype, so every site of a model decides alike: FP32 weights under BF16 autocast route
# per position for FP16, BF16 and FP32 features; FP16 weights keep the image routing
# (an FP16 RDUAL does not run run_window at all: its flow takes FP32 frames)
@needs_kernel
def test_routing_and_regional_na_follow_the_weights_not_the_features():
    conv, rca = _gpu_routed_conv(32), _gpu_regional(64)
    x = torch.randn(1, 64, 97, 40, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for half in (False, True):
            if half:
                conv.half()
                rca.half()
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                features = x.to(dtype)
                routes = conv.router(features[:, :32])
                assert (routes.shape[-2:] == (1, 1)) == half
                q, k, v = features, features, features
                rca.regional_na = False
                blend = rca.mix(q, k, v)
                rca.regional_na = True
                assert torch.equal(rca.mix(q, k, v), blend) == half
