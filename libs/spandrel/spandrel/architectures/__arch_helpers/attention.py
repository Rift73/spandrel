from __future__ import annotations

import torch
from torch.nn import functional as F


def pad_heads(x: torch.Tensor) -> torch.Tensor:
    """Attention heads (the last dimension) zero-padded to a multiple of 8 channels.

    scaled_dot_product_attention's fused kernels (cuDNN, memory-efficient) only take
    such head widths and fall back to the unfused kernel otherwise; HAT's and DAT's
    heads are often 30 wide (180 channels in 6 heads). The padding is exact: the zeros
    add nothing to q @ k^T, and the attention output's extra channels are dropped.
    """
    pad = -x.shape[-1] % 8
    return F.pad(x, (0, pad)) if pad else x
