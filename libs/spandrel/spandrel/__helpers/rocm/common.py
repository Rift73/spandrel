"""
When the custom HIP kernels may replace a module's computation, and the
depthwise 3x3 convolutions that use one.

Adapted from florentia-mysteria/spandrel-rocm (MIT).
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Callable, ClassVar

import torch
from torch import nn

from . import kernels

logger = logging.getLogger("spandrel.rocm")

# The number of active TorchDispatchModes. PyTorch has no public API for it;
# looked up by name because strict type checking rejects torch._C's members.
_dispatch_mode_count: Callable[[], int] = importlib.import_module(
    "torch._C"
)._len_torch_dispatch_stack


def version_of(t: torch.Tensor) -> int:
    """
    In-place modification counter, for cache keys. Inference tensors (e.g.
    weights converted to another dtype inside torch.inference_mode) have none,
    and they cannot be modified outside of inference mode anyway.
    """
    try:
        return t._version
    except RuntimeError:
        return -1


def _tracing() -> bool:
    """Whether a tracer, torch.compile or torch.export is recording the model."""
    # torch.jit.is_tracing is documented API but not in torch.jit.__all__, so
    # strict type checking rejects the attribute; is_exporting is newer than 2.4
    jit_is_tracing = getattr(torch.jit, "is_tracing", None)
    if jit_is_tracing is not None and jit_is_tracing():
        return True
    # not imported here: an export in progress has imported it
    onnx = sys.modules.get("torch.onnx")
    if onnx is not None and onnx.is_in_onnx_export():
        return True
    is_exporting = getattr(torch.compiler, "is_exporting", None)
    return torch.compiler.is_compiling() or (
        is_exporting is not None and is_exporting()
    )


def plain_inference(x: torch.Tensor, module: nn.Module) -> bool:
    """
    Whether a custom kernel may run module's computation on x: plain eager
    inference only. Tracing (torch.jit, ONNX export), torch.compile and
    torch.export, tensor subclasses (fake or functional tensors), training,
    autograd and autocast keep the original code, which they can see through.

    So does an active TorchDispatchMode: a mode sees every operation and may
    act on tensors between them (e.g. move an idle tensor to host memory),
    while a kernel reads raw pointers it cannot see.
    """
    if _tracing() or type(x) is not torch.Tensor or module.training:
        return False
    if _dispatch_mode_count() > 0:
        return False
    if torch.is_autocast_enabled(x.device.type):
        return False
    if torch.is_grad_enabled():
        return not x.requires_grad and not any(
            p.requires_grad for p in module.parameters()
        )
    return True


def is_depthwise_3x3(m: nn.Module) -> bool:
    """
    A plain nn.Conv2d (not a subclass, whose forward may differ) computing a
    depthwise 3x3 convolution with stride 1 and zero padding 1.
    """
    return (
        type(m) is nn.Conv2d
        and m.groups == m.in_channels == m.out_channels
        and m.kernel_size == (3, 3)
        and m.stride == (1, 1)
        and m.padding == (1, 1)
        and m.dilation == (1, 1)
        and m.padding_mode == "zeros"
    )


class HipDepthwiseConv2d(nn.Conv2d):
    """
    The class of the depthwise 3x3 convolutions patch_depthwise_convs routes
    through the HIP kernel. Anything the kernel cannot take runs nn.Conv2d's
    forward. Being a class (not a patched forward), it survives copy.deepcopy
    and pickling, and a copy uses its own weights.
    """

    # set for the rest of the process when the kernel fails
    kernel_failed: ClassVar[bool] = False

    # (key, fp32 weights in the kernel's (C, 9) layout, fp32 bias); the key
    # changes when the module moves to another device or dtype or its
    # parameters are replaced or modified in place. In-place edits through
    # `.data` are not detected: they do not change the version counter.
    _hip_weights: tuple[tuple[object, ...], torch.Tensor, torch.Tensor | None] | None
    _hip_weights = None

    def _get_name(self) -> str:
        return "Conv2d"

    def _weights_key(self) -> tuple[object, ...]:
        w, b = self.weight, self.bias
        key: tuple[object, ...] = (w.data_ptr(), w.dtype, w.device, version_of(w))
        if b is not None:
            key += (b.data_ptr(), version_of(b))
        return key

    def _kernel_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        key = self._weights_key()
        cached = self._hip_weights
        if cached is None or cached[0] != key:
            # built in full and stored in one assignment, so concurrent calls
            # never see half of an update
            w = self.weight.detach().float().reshape(self.in_channels, 9).contiguous()
            b = (
                self.bias.detach().float().contiguous()
                if self.bias is not None
                else None
            )
            cached = (key, w, b)
            self._hip_weights = cached
        return cached[1], cached[2]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (
            HipDepthwiseConv2d.kernel_failed
            or not kernels.hip_kernels_available()
            or not plain_inference(input, self)
            or not kernels.can_dwconv3x3(input, self.weight)
        ):
            return super().forward(input)
        w, b = self._kernel_weights()
        try:
            return kernels.dwconv3x3(input, w, b)
        except torch.cuda.OutOfMemoryError:
            raise
        except kernels.FAILURES as e:
            HipDepthwiseConv2d.kernel_failed = True
            logger.warning(
                "The HIP depthwise conv kernel failed; using the original "
                "convolution for the rest of this process. %s: %s",
                type(e).__name__,
                e,
            )
            logger.debug("HIP depthwise conv kernel failure", exc_info=True)
            return super().forward(input)


def patch_depthwise_convs(model: nn.Module) -> int:
    """
    Routes the model's depthwise 3x3 convolutions through a custom HIP kernel
    on ROCm, and returns how many there are. Does nothing elsewhere.

    MIOpen has no fast kernel for them on consumer Radeon cards: on an RX 9070 XT
    one takes ~4 ms on a 180x256x256 tensor, the custom kernel ~0.16 ms. DAT runs
    72 of them per tile, which made them its single biggest cost. Many other
    architectures (SAFMN, OmniSR, DRCT, RGT, ...) use them as well.

    Any failure (compile, launch, unsupported input) falls back to the original
    convolution. Set SPANDREL_HIP_KERNELS=0 to disable all custom HIP kernels.
    """
    if HipDepthwiseConv2d.kernel_failed or not kernels.hip_kernels_available():
        return 0
    count = 0
    for module in model.modules():
        if is_depthwise_3x3(module):
            module.__class__ = HipDepthwiseConv2d
            count += 1
    if count:
        logger.debug("Using the HIP kernel for %d depthwise 3x3 convolutions", count)
    return count
