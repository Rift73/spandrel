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


def window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None,
    mask: torch.Tensor | None,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """softmax(q @ k^T * scale + bias + mask) @ v over windows.

    q is (B*nW, heads, N, d), k and v (B*nW, heads, M, d), d a multiple of 8
    (pad_heads); bias is (1, heads, N, M) and mask (nW, N, M), the same for every image.

    Runs use scaled_dot_product_attention (cuDNN or memory-efficient attention), which
    takes bias and mask summed into one attn_mask. An ONNX export keeps the operations
    apart: TensorRT fuses them into its own attention kernel and adds bias and mask
    there, where their sum, (B*nW, heads, N, M), is a tensor as large as the attention
    matrix to write and read in every shifted block.
    """
    if torch.onnx.is_in_onnx_export():
        attn = (q * scale) @ k.transpose(-2, -1)
        if bias is not None:
            attn = attn + bias
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(-1, nw, *attn.shape[1:]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, *attn.shape[2:])
        return attn.softmax(dim=-1) @ v
    if mask is not None:
        nw = mask.shape[0]
        mask = mask.unsqueeze(1)
        bias = mask if bias is None else bias + mask
        bias = bias.unsqueeze(0).expand(q.shape[0] // nw, -1, -1, -1, -1).flatten(0, 1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, dropout_p=dropout_p, scale=scale
    )
