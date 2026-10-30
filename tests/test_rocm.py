"""
The ROCm HIP kernel paths (depthwise 3x3 convolution, ESRGAN's dense blocks).

Without an AMD GPU, the kernels are emulated: the launches go to an emulator
that runs each kernel's arithmetic with PyTorch on CPU memory, at exactly the
pointers, strides and offsets the launch passes. That checks everything on the
Python side (buffers, views, strides, launch arguments, guards and fallbacks).
The kernel sources are compiled with HIPRTC where one is available.
"""

from __future__ import annotations

import copy
import ctypes
import importlib
import logging
import os
import pickle
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn.functional import conv2d, leaky_relu
from torch.utils._python_dispatch import TorchDispatchMode

from spandrel import ImageModelDescriptor, ModelDescriptor
from spandrel.__helpers.rocm import kernels
from spandrel.__helpers.rocm.common import (
    HipDepthwiseConv2d,
    is_depthwise_3x3,
    patch_depthwise_convs,
)
from spandrel.architectures.__arch_helpers import block
from spandrel.architectures.DAT import DAT, DATArch
from spandrel.architectures.ESRGAN import ESRGAN, ESRGANArch
from spandrel.architectures.SAFMN import SAFMN, SAFMNArch

rrdb_path = importlib.import_module("spandrel.architectures.__arch_helpers._rocm_rrdb")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SUFFIX_DTYPE = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
DTYPE_SUFFIX = {dtype: suffix for suffix, dtype in SUFFIX_DTYPE.items()}
BUFFER = "_spandrel_rrdb_buffer"  # set by the RRDB path on its outputs


def _memory(address: int, count: int, dtype: torch.dtype) -> torch.Tensor:
    """count elements of dtype at a host address, sharing that memory."""
    size = count * torch.empty((), dtype=dtype).element_size()
    return torch.frombuffer(
        (ctypes.c_char * size).from_address(address), dtype=dtype, count=count
    )


class EmulatedHip:
    """Stands in for the HIP runtime: runs each launch with PyTorch on CPU memory."""

    def __init__(self) -> None:
        self.launched: list[str] = []
        self.attempts = 0
        self.fail: Exception | None = None
        # devices made current with torch.cuda.device, innermost last
        self.current: list[torch.device] = []
        # (kernel, current device) at every get and launch
        self.gets: list[tuple[str, torch.device | None]] = []
        self.launches: list[tuple[str, torch.device | None]] = []

    def current_device(self) -> torch.device | None:
        return self.current[-1] if self.current else None

    def get(self, device_index: int, program: str, name: str) -> str:
        assert program in kernels.PROGRAM_NAMES
        self.gets.append((name, self.current_device()))
        return name

    def launch(
        self,
        fn: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[
            ctypes.c_void_p | ctypes.c_int64 | ctypes.c_int | ctypes.c_float
        ],
        stream: int,
    ) -> None:
        self.attempts += 1
        self.launches.append((fn, self.current_device()))
        if self.fail is not None:
            raise self.fail
        kind, suffix = fn.rsplit("_", 1)
        values = [a.value for a in args]
        getattr(self, kind)(grid, block, SUFFIX_DTYPE[suffix], *values)
        self.launched.append(fn)

    @staticmethod
    def dwconv3x3(grid, block, dtype, x, w, b, y, c, h, width):
        assert block == (32, 16, 1)
        assert grid[:2] == ((width + 31) // 32, (h + 15) // 16)
        assert grid[2] % c == 0
        n = grid[2] * h * width
        image = _memory(x, n, dtype).view(-1, c, h, width)
        weight = _memory(w, c * 9, torch.float32).view(c, 1, 3, 3)
        bias = _memory(b, c, torch.float32) if b else None
        out = conv2d(image.float(), weight, bias, padding=1, groups=c)
        _memory(y, n, dtype).copy_(out.flatten())

    @staticmethod
    def act_into(grid, block, dtype, src, dst, n, dst_stride, dst_off, slope):
        size = torch.empty((), dtype=dtype).element_size()
        for b in range(grid[1]):
            s = _memory(src + b * n * size, n, dtype)
            d = _memory(dst + (b * dst_stride + dst_off) * size, n, dtype)
            d.copy_(leaky_relu(s, slope))

    @staticmethod
    def res_add(grid, block, dtype, a, x1, x1s, x2, x2s, dst, dsts, n, s1, s2, two):
        size = torch.empty((), dtype=dtype).element_size()
        for b in range(grid[1]):
            t = _memory(a + b * n * size, n, dtype) * s1
            t = t + _memory(x1 + b * x1s * size, n, dtype)
            if two:
                t = t * s2 + _memory(x2 + b * x2s * size, n, dtype)
            _memory(dst + b * dsts * size, n, dtype).copy_(t)


@pytest.fixture
def emulated_hip(monkeypatch: pytest.MonkeyPatch) -> EmulatedHip:
    hip = EmulatedHip()
    monkeypatch.setattr(kernels, "hip_kernels_available", lambda: True)
    monkeypatch.setattr(kernels, "_on_gpu", lambda t: t.device.type == "cpu")
    monkeypatch.setattr(kernels, "_stream", lambda t: 0)
    monkeypatch.setattr(kernels, "_get_hip", lambda: hip)
    monkeypatch.setattr(HipDepthwiseConv2d, "kernel_failed", False)
    monkeypatch.setattr(rrdb_path, "_failed", False)

    @contextmanager
    def current_device(device: torch.device) -> Iterator[None]:
        hip.current.append(device)
        try:
            yield
        finally:
            hip.current.pop()

    # stands in for torch.cuda.device, which only takes GPUs
    monkeypatch.setattr(torch.cuda, "device", current_device)
    return hip


@contextmanager
def kernels_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The original code path, as on a machine without ROCm."""
    with monkeypatch.context() as m:
        m.setattr(kernels, "hip_kernels_available", lambda: False)
        yield


def _rrdb_chain(nf: int, gc: int, blocks: int, dtype: torch.dtype) -> nn.Module:
    return (
        nn.Sequential(*[block.RRDB(nf, gc=gc) for _ in range(blocks)]).eval().to(dtype)
    )


# --- (a) ESRGAN's dense blocks on one buffer per block ------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(("nf", "gc"), [(64, 32), (13, 7)])
def test_rrdb_buffer_path_is_bit_identical(
    emulated_hip: EmulatedHip,
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    batch: int,
    nf: int,
    gc: int,
):
    model = _rrdb_chain(nf, gc, 2, dtype)
    x = (torch.randn(batch, nf, 11, 9) * 2).to(dtype)
    x_before = x.clone()

    reused: list[bool] = []
    original_input_buffer = rrdb_path._input_buffer

    def input_buffer(t: torch.Tensor, channels: int) -> torch.Tensor:
        buf = original_input_buffer(t, channels)
        reused.append(buf is getattr(t, BUFFER, None))
        return buf

    monkeypatch.setattr(rrdb_path, "_input_buffer", input_buffer)
    with torch.no_grad():
        fast = model(x)
        with kernels_off(monkeypatch):
            expected = model(x)

    # 2 RRDBs x 3 dense blocks x (4 activations + 1 residual)
    assert len(emulated_hip.launched) == 2 * 3 * 5
    # the second RRDB works in the buffer the first one wrote its output into
    assert reused == [False, True]
    assert torch.equal(x, x_before)
    assert fast.dtype == expected.dtype == dtype
    assert torch.equal(fast, expected)


def _grad_mode(name: str) -> torch.inference_mode | torch.no_grad:
    return torch.inference_mode() if name == "inference_mode" else torch.no_grad()


@pytest.mark.parametrize("grad_mode", ["no_grad", "inference_mode"])
def test_rrdb_reuses_only_its_own_buffers(
    emulated_hip: EmulatedHip, monkeypatch: pytest.MonkeyPatch, grad_mode: str
):
    model = _rrdb_chain(16, 8, 3, torch.float32)
    x = torch.randn(2, 16, 9, 7)
    inputs: list[torch.Tensor] = []
    copies: list[bool] = []
    original_input_buffer = rrdb_path._input_buffer

    def input_buffer(t: torch.Tensor, channels: int) -> torch.Tensor:
        buf = original_input_buffer(t, channels)
        inputs.append(t)
        copies.append(buf is not getattr(t, BUFFER, None))
        return buf

    monkeypatch.setattr(rrdb_path, "_input_buffer", input_buffer)
    with _grad_mode(grad_mode):
        fast = model(x)
        with kernels_off(monkeypatch):
            expected = model(x)
    # views made in inference mode have no _base: the mark is an attribute
    assert copies == [True, False, False]
    assert all(getattr(t, BUFFER, None) is not None for t in inputs[1:])
    assert getattr(fast, BUFFER).shape == (2, 16 + 4 * 8, 9, 7)
    assert torch.equal(fast, expected)


@pytest.mark.parametrize("grad_mode", ["no_grad", "inference_mode"])
def test_rrdb_never_writes_into_a_callers_buffer(
    emulated_hip: EmulatedHip, monkeypatch: pytest.MonkeyPatch, grad_mode: str
):
    nf, gc = 16, 8
    model = _rrdb_chain(nf, gc, 1, torch.float32)
    with _grad_mode(grad_mode):
        # the caller's own buffer of the size the RRDB path works in
        own = torch.randn(1, nf + 4 * gc, 8, 8)
        before = own.clone()
        fast = model(own[:, :nf])
        with kernels_off(monkeypatch):
            expected = model(own[:, :nf])
    assert len(emulated_hip.launched) == 15
    assert torch.equal(own, before)
    assert torch.equal(fast, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 2])
def test_esrgan_on_the_buffer_path_is_bit_identical(
    emulated_hip: EmulatedHip,
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    batch: int,
):
    descriptor = ESRGANArch().load(
        ESRGAN(in_nc=3, out_nc=3, num_filters=64, num_blocks=3, scale=2).state_dict()
    )
    model = descriptor.model.eval().to(dtype)
    x = torch.rand(batch, 3, 10, 12).to(dtype)
    with torch.no_grad():
        fast = model(x)
        with kernels_off(monkeypatch):
            expected = model(x)
    assert len(emulated_hip.launched) == 3 * 3 * 5
    assert torch.equal(fast, expected)


def test_rrdb_variants_keep_the_original_code(emulated_hip: EmulatedHip):
    x = torch.randn(1, 16, 8, 8)
    with torch.no_grad():
        for rrdb in [
            block.RRDB(16, gc=8, plus=True),  # ESRGAN+ residual paths
            block.RRDB(16, gc=8, act_type="relu"),
            block.RRDB(16, gc=8, act_type="prelu"),
            block.RRDB(16, gc=8, norm_type="batch"),
            block.RRDB(16, gc=8, pad_type="reflect"),
        ]:
            rrdb.eval()(x)
        # a dtype the weights do not have: the original code reports it
        with pytest.raises(RuntimeError):
            block.RRDB(16, gc=8).eval()(x.half())
    assert emulated_hip.launched == []


# --- (b) depthwise 3x3 convolutions ---------------------------------------------------


class SubclassedConv(nn.Conv2d):
    pass


def _conv_zoo(c: int = 6) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "dw": nn.Conv2d(c, c, 3, padding=1, groups=c),
            "dw_no_bias": nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            "dense": nn.Conv2d(c, c, 3, padding=1),
            "dw_5x5": nn.Conv2d(c, c, 5, padding=2, groups=c),
            "dw_stride": nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c),
            "dw_dilated": nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c),
            "dw_reflect": nn.Conv2d(
                c, c, 3, padding=1, groups=c, padding_mode="reflect"
            ),
            "dw_multiplier": nn.Conv2d(c, 2 * c, 3, padding=1, groups=c),
            "dw_unpadded": nn.Conv2d(c, c, 3, groups=c),
            "subclass": SubclassedConv(c, c, 3, padding=1, groups=c),
        }
    ).eval()


def _conv(name: str) -> nn.Conv2d:
    conv = _conv_zoo()[name]
    assert isinstance(conv, nn.Conv2d)
    return conv


def test_only_depthwise_3x3_convs_are_patched(emulated_hip: EmulatedHip):
    zoo = _conv_zoo()
    matching = {name for name, m in zoo.items() if is_depthwise_3x3(m)}
    assert matching == {"dw", "dw_no_bias"}
    assert patch_depthwise_convs(zoo) == 2
    patched = {n for n, m in zoo.items() if isinstance(m, HipDepthwiseConv2d)}
    assert patched == matching
    assert type(zoo["subclass"]) is SubclassedConv
    assert repr(zoo["dw"]).startswith("Conv2d(6, 6, kernel_size=(3, 3)")
    # patching again changes nothing
    assert patch_depthwise_convs(zoo) == 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("name", ["dw", "dw_no_bias"])
def test_patched_conv_equals_the_conv(
    emulated_hip: EmulatedHip, dtype: torch.dtype, batch: int, name: str
):
    conv = _conv(name)
    patch_depthwise_convs(conv)
    conv.to(dtype)
    x = torch.randn(batch, 6, 19, 37).to(dtype)
    with torch.no_grad():
        fast = conv(x)
        expected = nn.Conv2d.forward(conv, x)
    assert emulated_hip.launched == [f"dwconv3x3_{DTYPE_SUFFIX[dtype]}"]
    assert fast.dtype == dtype and fast.is_contiguous()
    if dtype == torch.float32:
        assert torch.equal(fast, expected)
    else:
        torch.testing.assert_close(fast.float(), expected.float(), rtol=0.02, atol=0.02)


def test_patched_conv_layouts(emulated_hip: EmulatedHip):
    conv = _conv("dw")
    patch_depthwise_convs(conv)
    x = torch.randn(2, 12, 9, 10)
    with torch.no_grad():
        # a channel slice of a batch: not contiguous, still NCHW
        part = x[:, 3:9]
        assert torch.equal(conv(part), nn.Conv2d.forward(conv, part))
        assert len(emulated_hip.launched) == 1
        # channels-last keeps the original conv, which keeps the memory format
        cl = x[:, :6].contiguous(memory_format=torch.channels_last)
        out = conv(cl)
        assert out.is_contiguous(memory_format=torch.channels_last)
        assert len(emulated_hip.launched) == 1
        # wrong channel count: the original conv reports it
        with pytest.raises(RuntimeError):
            conv(x)
        assert len(emulated_hip.launched) == 1


def test_weight_cache_follows_the_module(emulated_hip: EmulatedHip):
    conv = _conv("dw")
    patch_depthwise_convs(conv)
    x = torch.randn(1, 6, 8, 8)
    with torch.no_grad():
        first = conv(x)
        conv.weight.mul_(2)  # in place
        assert not torch.equal(conv(x), first)
        assert torch.equal(conv(x), nn.Conv2d.forward(conv, x))
        assert conv.bias is not None
        conv.bias.add_(1)
        assert torch.equal(conv(x), nn.Conv2d.forward(conv, x))
        conv.half()  # new parameters of another dtype
        torch.testing.assert_close(
            conv(x.half()).float(),
            nn.Conv2d.forward(conv, x.half()).float(),
            rtol=0.02,
            atol=0.02,
        )
        conv.float()
        conv.load_state_dict({"weight": torch.ones(6, 1, 3, 3), "bias": torch.zeros(6)})
        assert torch.equal(conv(x), nn.Conv2d.forward(conv, x))


def test_patched_conv_copies_and_pickles(emulated_hip: EmulatedHip):
    conv = _conv("dw")
    patch_depthwise_convs(conv)
    x = torch.randn(1, 6, 8, 8)
    with torch.no_grad():
        conv(x)
        copied = copy.deepcopy(conv)
        copied.weight.mul_(3)
        assert isinstance(copied, HipDepthwiseConv2d)
        assert torch.equal(copied(x), nn.Conv2d.forward(copied, x))
        assert torch.equal(conv(x), nn.Conv2d.forward(conv, x))
        restored = pickle.loads(pickle.dumps(conv))
        assert torch.equal(restored(x), conv(x))


def test_models_route_their_depthwise_convs_at_load(
    emulated_hip: EmulatedHip, monkeypatch: pytest.MonkeyPatch
):
    torch.manual_seed(0)
    models: list[ModelDescriptor] = [
        SAFMNArch().load(
            SAFMN(dim=8, n_blocks=3, ffn_scale=5.0, upscaling_factor=2).state_dict()
        ),
        DATArch().load(
            DAT(embed_dim=24, depth=[2, 2], num_heads=[2, 2], upscale=2).state_dict()
        ),
    ]
    x = torch.rand(1, 3, 16, 16)
    for descriptor in models:
        assert isinstance(descriptor, ImageModelDescriptor)
        count = sum(
            isinstance(m, HipDepthwiseConv2d) for m in descriptor.model.modules()
        )
        assert count > 0
        emulated_hip.launched.clear()
        with torch.no_grad():
            fast = descriptor(x)
            with kernels_off(monkeypatch):
                expected = descriptor(x)
        assert len(emulated_hip.launched) >= count
        torch.testing.assert_close(fast, expected, rtol=1e-5, atol=1e-5)


# --- (c) nothing changes without ROCm ---------------------------------------------------


def test_nothing_is_patched_or_launched_without_rocm(monkeypatch: pytest.MonkeyPatch):
    if kernels.hip_kernels_available():
        pytest.skip("a ROCm GPU is present")

    def no_hip():
        raise AssertionError("the HIP runtime was used")

    monkeypatch.setattr(kernels, "_get_hip", no_hip)
    esrgan = ESRGANArch().load(
        ESRGAN(in_nc=3, out_nc=3, num_filters=16, num_blocks=2, scale=2).state_dict()
    )
    safmn = SAFMNArch().load(
        SAFMN(dim=8, n_blocks=3, ffn_scale=5.0, upscaling_factor=2).state_dict()
    )
    for descriptor in (esrgan, safmn):
        assert not any(
            isinstance(m, HipDepthwiseConv2d) for m in descriptor.model.modules()
        )
        assert isinstance(descriptor, ImageModelDescriptor)
        with torch.no_grad():
            descriptor(torch.rand(1, 3, 8, 8))
    assert patch_depthwise_convs(_conv_zoo()) == 0


def test_availability(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("SPANDREL_HIP_KERNELS", raising=False)

    monkeypatch.setattr(torch.version, "hip", None)  # CPU-only or CUDA build
    assert not kernels.hip_kernels_available()

    monkeypatch.setattr(torch.version, "hip", "7.14.60850")
    for platform, available in [("win32", True), ("linux", True), ("darwin", False)]:
        monkeypatch.setattr(sys, "platform", platform)
        assert kernels.hip_kernels_available() == available
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SPANDREL_HIP_KERNELS", "0")
    assert not kernels.hip_kernels_available()
    monkeypatch.setenv("SPANDREL_HIP_KERNELS", "1")
    assert kernels.hip_kernels_available()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not kernels.hip_kernels_available()


# --- (d) fallback after a failure -------------------------------------------------------


def test_rrdb_falls_back_for_the_rest_of_the_process(
    emulated_hip: EmulatedHip,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    model = _rrdb_chain(16, 8, 1, torch.float32)
    x = torch.randn(1, 16, 8, 8)
    with torch.no_grad(), kernels_off(monkeypatch):
        expected = model(x)

    emulated_hip.fail = kernels.HipKernelError("simulated launch failure")
    with caplog.at_level(logging.WARNING, logger="spandrel.rocm"), torch.no_grad():
        assert torch.equal(model(x), expected)
        assert torch.equal(model(x), expected)
    assert emulated_hip.attempts == 1
    assert rrdb_path._failed
    warnings = [r for r in caplog.records if r.name == "spandrel.rocm"]
    assert len(warnings) == 1
    assert "HipKernelError: simulated launch failure" in warnings[0].getMessage()


def test_dwconv_falls_back_for_the_rest_of_the_process(
    emulated_hip: EmulatedHip, caplog: pytest.LogCaptureFixture
):
    conv = _conv("dw")
    patch_depthwise_convs(conv)
    x = torch.randn(1, 6, 8, 8)
    emulated_hip.fail = kernels.HipKernelError("simulated compile failure")
    with caplog.at_level(logging.WARNING, logger="spandrel.rocm"), torch.no_grad():
        assert torch.equal(conv(x), nn.Conv2d.forward(conv, x))
        assert torch.equal(conv(x), nn.Conv2d.forward(conv, x))
    assert emulated_hip.attempts == 1
    assert HipDepthwiseConv2d.kernel_failed
    assert "simulated compile failure" in caplog.records[0].getMessage()
    # nothing new is routed once the kernel failed
    assert patch_depthwise_convs(_conv_zoo()) == 0


def test_out_of_memory_is_raised_not_a_kernel_failure(emulated_hip: EmulatedHip):
    conv = _conv("dw")
    patch_depthwise_convs(conv)
    rrdb = _rrdb_chain(16, 8, 1, torch.float32)
    emulated_hip.fail = torch.cuda.OutOfMemoryError("simulated out of memory")
    with torch.no_grad():
        with pytest.raises(torch.cuda.OutOfMemoryError):
            conv(torch.randn(1, 6, 8, 8))
        with pytest.raises(torch.cuda.OutOfMemoryError):
            rrdb(torch.randn(1, 16, 8, 8))
    assert not HipDepthwiseConv2d.kernel_failed
    assert not rrdb_path._failed


# --- (e) export, tracing, compiling, training: the original code -----------------------


def _models() -> tuple[nn.Module, nn.Module]:
    conv = _conv("dw")
    patch_depthwise_convs(conv)
    return conv, _rrdb_chain(16, 8, 1, torch.float32)


def test_tracing_records_the_original_code(emulated_hip: EmulatedHip):
    for model, channels in zip(_models(), (6, 16)):
        x = torch.randn(1, channels, 8, 8)
        with torch.no_grad():
            # (check_trace would also run the model eagerly, on the kernels)
            traced = torch.jit.trace(model, x, check_trace=False)
            assert isinstance(traced, torch.jit.ScriptModule)
            assert emulated_hip.launched == []
            graph = str(traced.inlined_graph)
            assert "aten::_convolution" in graph
            assert "aten::empty" not in graph
            assert torch.equal(traced(x), model(x))
            emulated_hip.launched.clear()


@pytest.mark.parametrize(
    "state", ["onnx_export", "compiling", "exporting", "training", "grad", "autocast"]
)
def test_fast_paths_are_skipped(
    emulated_hip: EmulatedHip, monkeypatch: pytest.MonkeyPatch, state: str
):
    if state == "onnx_export":
        onnx = importlib.import_module("torch.onnx")
        monkeypatch.setattr(onnx, "is_in_onnx_export", lambda: True)
    elif state == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif state == "exporting":
        monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True)
    for model, channels in zip(_models(), (6, 16)):
        x = torch.randn(1, channels, 8, 8)
        if state == "training":
            model.train()(x)
        elif state == "grad":
            model(x.requires_grad_())
            model(x.detach())  # the parameters require grad
        elif state == "autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
                model(x)
        else:
            with torch.no_grad():
                model(x)
    assert emulated_hip.launched == []


class PassThroughMode(TorchDispatchMode):
    """A dispatch mode that sees every operation and changes nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops += 1
        return func(*args, **(kwargs or {}))


def test_dispatch_modes_keep_the_original_code(
    emulated_hip: EmulatedHip, monkeypatch: pytest.MonkeyPatch
):
    for model, channels in zip(_models(), (6, 16)):
        x = torch.randn(1, channels, 8, 8)
        with torch.no_grad():
            with kernels_off(monkeypatch):
                expected = model(x)
            mode = PassThroughMode()
            with mode:
                actual = model(x)
        assert mode.ops > 0  # the original operations ran under the mode
        assert torch.equal(actual, expected)
    assert emulated_hip.launched == []
    assert emulated_hip.attempts == 0


def test_kernels_launch_on_the_tensors_device(emulated_hip: EmulatedHip):
    # The default stream (handle 0) is the null stream of the current device,
    # so the tensor's device must be current at every launch (and compile).
    for model, channels in zip(_models(), (6, 16)):
        x = torch.randn(1, channels, 8, 8)
        with torch.no_grad():
            model(x)
    assert len(emulated_hip.launches) == 1 + 15
    assert emulated_hip.gets == emulated_hip.launches
    assert all(device == torch.device("cpu") for _, device in emulated_hip.launches)
    assert emulated_hip.current == []


def test_tensor_subclasses_keep_the_original_code(emulated_hip: EmulatedHip):
    class Wrapped(torch.Tensor):
        pass

    for model, channels in zip(_models(), (6, 16)):
        with torch.no_grad():
            model(torch.randn(1, channels, 8, 8).as_subclass(Wrapped))
    assert emulated_hip.launched == []


def test_eager_inference_uses_the_kernels(emulated_hip: EmulatedHip):
    # the counterpart of the skips above
    for model, channels in zip(_models(), (6, 16)):
        with torch.inference_mode():
            model(torch.randn(1, channels, 8, 8))
        with torch.no_grad():
            model(torch.randn(1, channels, 8, 8))
    assert len(emulated_hip.launched) == 2 * (1 + 15)


# --- library discovery, cache directory, compiling ----------------------------------------


def test_library_kinds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert kernels.library_kind("amdhip64_7.dll") == "hip"
    assert kernels.library_kind("hiprtc0714.dll") == "hiprtc"
    assert kernels.library_kind("hiprtc-builtins0714.dll") is None
    assert kernels.library_kind("amdhip64_7.lib") is None
    monkeypatch.setattr(sys, "platform", "linux")
    assert kernels.library_kind("libamdhip64.so.7") == "hip"
    assert kernels.library_kind("libamdhip64.so") == "hip"
    assert kernels.library_kind("libhiprtc.so.7.14.60850") == "hiprtc"
    assert kernels.library_kind("libhiprtc-builtins.so.7") is None
    assert kernels.library_kind("libc10_hip.so") is None
    assert kernels.library_kind("amdhip64_7.dll") is None


def test_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("SPANDREL_HIP_CACHE_DIR", str(tmp_path))
    assert kernels.cache_dir() == tmp_path
    monkeypatch.setenv("SPANDREL_HIP_CACHE_DIR", "")
    assert kernels.cache_dir() is None
    monkeypatch.delenv("SPANDREL_HIP_CACHE_DIR")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert kernels.cache_dir() == tmp_path / "spandrel" / "hip_cache"

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert kernels.cache_dir() == tmp_path / "spandrel" / "hip_cache"
    for xdg in ("", "relative/cache"):
        monkeypatch.setenv("XDG_CACHE_HOME", xdg)
        expected = Path.home() / ".cache" / "spandrel" / "hip_cache"
        assert kernels.cache_dir() == expected


@pytest.mark.skipif(
    not torch.version.hip, reason="needs a ROCm build of PyTorch (HIP runtime loaded)"
)
def test_finds_the_hip_runtime_pytorch_loaded():
    hip, rtc, runtime = kernels.open_hip_libraries()
    assert kernels.library_kind(runtime.name) == "hip"
    assert hip.hipModuleLaunchKernel is not None
    assert kernels.hiprtc_version(rtc)
    if sys.platform == "linux":
        maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
        assert str(runtime) in maps


def _hiprtc() -> Path | None:
    """SPANDREL_TEST_HIPRTC, or the HIPRTC of the installed ROCm Python packages."""
    path = os.environ.get("SPANDREL_TEST_HIPRTC")
    if path:
        return Path(path)
    try:
        rocm_sdk = importlib.import_module("rocm_sdk")
    except ImportError:
        return None
    found = rocm_sdk.find_libraries("hiprtc")
    return Path(found[0]) if found else None


HIPRTC = _hiprtc()
TARGETS = ["gfx1201", "gfx1200", "gfx1100", "gfx1101"]
TARGETS += ["gfx1030", "gfx942", "gfx90a", "gfx908"]


@pytest.mark.skipif(
    HIPRTC is None,
    reason="no HIPRTC (set SPANDREL_TEST_HIPRTC or install ROCm's Python packages)",
)
@pytest.mark.parametrize("arch", TARGETS)
@pytest.mark.parametrize("program", kernels.PROGRAM_NAMES)
def test_kernels_compile(program: str, arch: str):
    assert HIPRTC is not None
    code = kernels.compile_program(kernels.open_hiprtc(HIPRTC), program, arch)
    # an ELF code object, or an offload bundle of one
    assert code.startswith((b"\x7fELF", b"__CLANG_OFFLOAD_BUNDLE__"))


def test_compile_errors_carry_the_log():
    if HIPRTC is None:
        pytest.skip("no HIPRTC")
    with pytest.raises(kernels.HipKernelError, match="offload-arch|gfx0"):
        kernels.compile_program(kernels.open_hiprtc(HIPRTC), "rrdb", "gfx0")
