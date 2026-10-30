"""
ROCm acceleration controls.

On ROCm builds of PyTorch (Windows and Linux) with an AMD GPU, spandrel runs
depthwise 3x3 convolutions and ESRGAN's dense blocks on custom HIP kernels,
compiled at runtime with HIPRTC, which ships with ROCm. The kernels are adapted
from florentia-mysteria/spandrel-rocm (MIT). Elsewhere nothing changes.

Environment variables:

- SPANDREL_HIP_KERNELS=0 disables all custom kernels. It is read on every
  call; depthwise convolutions are routed when a model is loaded, so a model
  loaded while it was 0 keeps its original convolutions.
- SPANDREL_HIP_CACHE_DIR sets where compiled kernels are cached (empty: no
  cache). Default: spandrel/hip_cache in %LOCALAPPDATA% on Windows and in
  $XDG_CACHE_HOME (default ~/.cache) on Linux.

Any failure (library not found, compile, launch) is logged as a warning on the
"spandrel.rocm" logger, and the affected operation runs its original code for
the rest of the process.
"""

from __future__ import annotations

from .__helpers.rocm.kernels import hip_kernels_available as is_available

__all__ = ["is_available"]
