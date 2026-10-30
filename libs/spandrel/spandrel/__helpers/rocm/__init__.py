"""
ROCm acceleration: custom HIP kernels for operations that PyTorch and MIOpen
handle poorly on consumer Radeon GPUs, and the code that routes models to them.

Adapted from florentia-mysteria/spandrel-rocm (MIT, the same license and
copyright notice as spandrel's own LICENSE): the depthwise 3x3 convolution and
ESRGAN's dense-block kernels, extended to Linux.

Everything here is optional. Without a ROCm build of PyTorch with a GPU (or with
SPANDREL_HIP_KERNELS=0), every model runs its original code.
"""
