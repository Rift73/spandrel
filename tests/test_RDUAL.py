import importlib
import math
import os
import subprocess
import sys
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
