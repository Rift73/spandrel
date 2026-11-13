# HEART - Hybrid Efficient Attention with Rank-factorized bias Transformer
# based on HAT-iLN (arXiv:2504.06629) and SST's RIB (arXiv:2603.06738)
#
# Source: huggingface.co/Phips/HEART, heart_arch.py (SHA-256
# 7b7788cdcf2c3acd61f5b601e291ed7cfdc20c06153aa3c9728a3473380c14cf), Apache-2.0.
# spandrel: the traiNNer registration and the `heart` factory are not bundled
# (they would collide with traiNNer's copy when both are imported). The layers
# the original imports from traiNNer's hat_iln_arch and arch_util are bundled
# below; i-LN uses the eps of the published ONNX export (1e-4). The `norm_layer`
# argument is not a constructor argument here: the blocks unpack i-LN's
# (x, std) pair, so i-LN is the only norm they accept.
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from spandrel.__helpers.rocm.common import plain_inference
from spandrel.util import store_hyperparameters
from spandrel.util.timm import to_2tuple, trunc_normal_

from ...__arch_helpers.flash_window import flash_window_attention
from ...__arch_helpers.inference import (
    Derived,
    banded_tail,
    tail_scale_and_halo,
    traced,
)
from .window_triton import heart_window_attention

# spandrel: the high-resolution tail (conv_before_upsample, upsample, conv_last) runs in
# bands of a BAND_DIVISOR-th of the padded trunk's rows, at least MIN_BAND_ROWS, as ESRGAN's
# does; a trunk of WHOLE_TAIL_ROWS rows or fewer runs it whole.
BAND_DIVISOR = 8
MIN_BAND_ROWS = 32
WHOLE_TAIL_ROWS = 256


def token_layout(x: torch.Tensor, module: nn.Module) -> bool:
    """spandrel: whether module runs on x in the token layout, (B, H, W, C) memory with
    channels-last convolutions on views of it and 1x1 convolutions as linear maps:
    plain eager inference on an NVIDIA GPU. The CPU, ROCm, tracing and training keep the
    original layout."""
    return (
        x.is_cuda
        and torch.version.hip is None
        and not traced()
        and plain_inference(x, module)
    )


def _reflect_index(n: int, pad: int, device: torch.device) -> torch.Tensor:
    """The rows (or columns) of a reflect padding of n by pad on each side."""
    i = torch.arange(-pad, n + pad, device=device)
    return torch.where(i < 0, -i, torch.where(i >= n, 2 * (n - 1) - i, i))


class iLN(nn.Module):
    """Image restoration tailored layer normalization (i-LN).

    Normalizes across both the spatial and the channel dimensions instead of
    per token, and returns the standard deviation for the residual rescale.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C) where L = H*W
        b = x.shape[0]
        if x.is_cuda and torch.version.hip is None and not traced():
            if plain_inference(x, self):
                return self.fused(x)
        x_flat = x.reshape(b, -1)
        mean = x_flat.mean(dim=1, keepdim=True).reshape(b, 1, 1)
        var = x_flat.var(dim=1, keepdim=True, unbiased=False).reshape(b, 1, 1)
        std = torch.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        return self.weight * x_norm + self.bias, std

    def fused(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """spandrel: eager CUDA inference in two passes over x instead of six: one
        reduction for mean and variance, then x * (weight / std) + (bias - mean *
        weight / std) in one kernel."""
        b = x.shape[0]
        var, mean = torch.var_mean(x.reshape(b, -1), dim=1, keepdim=True, correction=0)
        std = torch.sqrt(var + self.eps).reshape(b, 1, 1)
        scale = self.weight / std
        shift = self.bias - mean.reshape(b, 1, 1) * scale
        return torch.addcmul(shift, x, scale), std


class AffineTransform(nn.Module):
    """gamma * x + beta without normalization (the final norm of i-LN networks)."""

    def __init__(self, normalized_shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat: int, squeeze_factor: int = 16) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(
        self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30
    ) -> None:
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """(B, C, H, W) -> (B, H*W, C), with an optional affine norm."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        embed_dim: int = 180,
        patch_norm: bool = False,
    ) -> None:
        super().__init__()
        img = to_2tuple(img_size)
        patch = to_2tuple(patch_size)
        self.patches_resolution = [img[0] // patch[0], img[1] // patch[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.norm = AffineTransform(embed_dim) if patch_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """(B, H*W, C) -> (B, C, H, W)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        )


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        m: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super().__init__(*m)


class RIBWindowAttention(nn.Module):
    """Window attention with Rank-factorized Implicit Neural Bias (RIB).

    RIB (SST, arXiv 2603.06738) replaces the relative position bias table:
    position features from a small implicit net are concatenated onto Q/K,
    turning bias addition into a dot product, so attention runs on
    FlashAttention/SDPA kernels with no bias table and no masks.
    Channels-first (B, C, H, W); shifted windows use non-wrapping
    pad-and-partition (no cyclic wrap).
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        shift: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

        self.rank = rank

        # normalized window coordinates
        wh, ww = self.window_size
        yy, xx = torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()  # (N, 2)
        coords[:, 0] = (2.0 * (coords[:, 0] + 0.5) / ww) - 1.0
        coords[:, 1] = (2.0 * (coords[:, 1] + 0.5) / wh) - 1.0

        self.n_freqs = rib_n_freqs
        if self.n_freqs > 0:
            base_coords = coords.clone()
            for i in range(self.n_freqs):
                freq = 2**i
                coords = torch.cat(
                    [
                        coords,
                        torch.sin(base_coords * freq),
                        torch.cos(base_coords * freq),
                    ],
                    dim=-1,
                )
        self.rib_coords: torch.Tensor
        self.register_buffer("rib_coords", coords, persistent=False)

        n_input = 2 + 4 * self.n_freqs
        hidden_d = rib_hidden_dim
        self.to_hidden = nn.Parameter(torch.empty(n_input, hidden_d))
        self.hidden_b = nn.Parameter(torch.zeros(1, hidden_d))
        self.to_q = nn.Parameter(torch.empty(num_heads, hidden_d, self.rank))
        self.to_k = nn.Parameter(torch.empty(num_heads, hidden_d, self.rank))

        nn.init.normal_(self.to_hidden, mean=0.0, std=0.05)
        nn.init.normal_(self.to_q, mean=0.0, std=0.05)
        nn.init.normal_(self.to_k, mean=0.0, std=0.05)

        self.shift = shift

    def pad_to_win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # inputs are pre-aligned to window multiples by HEART.check_image_size
        # (which owns the replicate fallback), so reflect padding is safe here
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    @staticmethod
    def _cat_position(
        q: torch.Tensor,
        k: torch.Tensor,
        pos_q: torch.Tensor,
        pos_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: (Bwin, N, heads, D); pos_q, pos_k: (Bwin, heads, N, R)
        d = q.shape[-1]
        pos_q = pos_q.transpose(1, 2)  # (Bwin, N, heads, R)
        pos_k = pos_k.transpose(1, 2)
        r = pos_q.shape[-1]

        q = q * (d**-0.5)
        pos_q = pos_q * (r**-0.5)
        return torch.cat([q, pos_q], dim=-1), torch.cat([k, pos_k], dim=-1)

    def _flash_cat_attn(
        self, q_cat: torch.Tensor, k_cat: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # q_cat, k_cat: (Bwin, N, heads, D + R); v: (Bwin, N, heads, D)
        d = v.shape[-1]
        r = q_cat.shape[-1] - d

        # spandrel: eager inference runs it in flash_window's kernel when it can,
        # which takes q, k and v as they are; scaled_dot_product_attention's fused
        # kernels want widths of a multiple of 8 and v as wide as q and k.
        if not self.training:
            out = flash_window_attention(
                q_cat.transpose(1, 2),
                k_cat.transpose(1, 2),
                v.transpose(1, 2),
                None,
                1.0,
            )
            if out is not None:
                return out.transpose(1, 2)

        d_total = q_cat.shape[-1]
        pad = (8 - (d_total % 8)) % 8
        if pad:
            q_cat = F.pad(q_cat, (0, pad))
            k_cat = F.pad(k_cat, (0, pad))
        v_cat = F.pad(v, (0, r + pad))

        # q and pos_q are pre-scaled (SST eq. 5); scale=1.0 keeps SDPA from
        # rescaling them again
        return F.scaled_dot_product_attention(
            q_cat.transpose(1, 2),
            k_cat.transpose(1, 2),
            v_cat.transpose(1, 2),
            scale=1.0,
        ).transpose(1, 2)[:, :, :, :d]

    def _positions(
        self, bwin: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # RIB position features (fp32 for fp16-inference stability)
        intermediate = F.relu(
            self.rib_coords.to(dtype=torch.float32)
            @ self.to_hidden.to(dtype=torch.float32)
            + self.hidden_b.to(dtype=torch.float32)
        )  # (N, hidden_d)
        q_pos = torch.einsum(
            "nd,hdr->hnr", intermediate, self.to_q.to(dtype=torch.float32)
        )  # (heads, N, R)
        k_pos = torch.einsum(
            "nd,hdr->hnr", intermediate, self.to_k.to(dtype=torch.float32)
        )
        q_pos = (
            q_pos.to(dtype).unsqueeze(0).expand(bwin, -1, -1, -1)
        )  # (Bwin, heads, N, R)
        k_pos = k_pos.to(dtype).unsqueeze(0).expand(bwin, -1, -1, -1)
        return q_pos, k_pos

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """spandrel: the attention on tokens x (B, H, W, C), H and W multiples of the
        window, as (B, H, W, C). The projections are linear maps on the tokens; one
        gather pads (shifted windows) and partitions each of q, k and v, and one
        gathers the windows back and crops."""
        b, h, w, _ = x.shape
        ws = self.window_size[0]
        s = ws // 2 if self.shift else 0
        hd, wd = (h + 2 * s) // ws, (w + 2 * s) // ws
        rows = _reflect_index(h, s, x.device).view(hd, 1, ws, 1)
        cols = _reflect_index(w, s, x.device).view(1, wd, 1, ws)
        qkv = F.linear(x, self.to_qkv.weight.flatten(1), self.to_qkv.bias)
        # the attention's width, which a copy with a wider trunk keeps
        c = qkv.shape[-1] // 3
        pos_q, pos_k = self._positions(1, qkv.dtype)
        d, r = c // self.num_heads, pos_q.shape[-1]
        out = heart_window_attention(
            qkv,
            (pos_q[0] * r**-0.5).contiguous(),
            pos_k[0].contiguous(),
            self.num_heads,
            ws,
            s,
            d**-0.5,
        )
        if out is not None:
            return F.linear(out, self.to_out.weight.flatten(1), self.to_out.bias)
        q, k, v = (
            qkv[..., i * c : (i + 1) * c][:, rows, cols].view(
                b * hd * wd, ws * ws, self.num_heads, -1
            )
            for i in range(3)
        )  # (Bwin, N, heads, D)
        del qkv
        q_pos, k_pos = self._positions(q.shape[0], q.dtype)
        q_cat, k_cat = self._cat_position(q, k, q_pos, k_pos)
        del q, k
        out = self._flash_cat_attn(q_cat, k_cat, v)  # (Bwin, N, heads, D)
        del q_cat, k_cat, v
        # each token's window and place in it, past the shift's padding
        ys = torch.arange(s, h + s, device=x.device)
        xs = torch.arange(s, w + s, device=x.device)
        out = out.reshape(b, hd, wd, ws, ws, c)[
            :,
            (ys // ws)[:, None],
            (xs // ws)[None, :],
            (ys % ws)[:, None],
            (xs % ws)[None, :],
        ]
        return F.linear(out, self.to_out.weight.flatten(1), self.to_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)

        # shifted windows: pad top/left by half a window, partition from the
        # padded origin. x is window-aligned here, so padding (sh, sh) per
        # axis keeps alignment (one extra window row/col).
        sh = sw = 0
        if self.shift:
            sh, sw = self.window_size[0] // 2, self.window_size[1] // 2
            x = F.pad(x, (sw, sw, sh, sh), mode="reflect")

        h_div, w_div = (
            x.shape[2] // self.window_size[0],
            x.shape[3] // self.window_size[1],
        )

        qkv = self.to_qkv(x)
        del x
        # spandrel: q, k and v in tensors of their own, so that q's and k's memory
        # is freed once their position features are attached
        q, k, v = (
            rearrange(
                part,
                "b (heads c) (h wh) (w ww) -> (b h w) (wh ww) heads c",
                heads=self.num_heads,
                wh=self.window_size[0],
                ww=self.window_size[1],
            )
            for part in qkv.chunk(3, dim=1)
        )  # (Bwin, N, heads, D)
        del qkv

        q_pos, k_pos = self._positions(q.shape[0], q.dtype)
        q_cat, k_cat = self._cat_position(q, k, q_pos, k_pos)
        del q, k
        out = self._flash_cat_attn(q_cat, k_cat, v)  # (Bwin, N, heads, D)
        del q_cat, k_cat, v

        out = rearrange(
            out,
            "(b h w) (wh ww) heads c -> b (heads c) (h wh) (w ww)",
            h=h_div,
            w=w_div,
            wh=self.window_size[0],
            ww=self.window_size[1],
        )
        out = out[:, :, sh : sh + h, sw : sw + w]
        return self.to_out(out)


class HAB_RIB(Derived):
    """Hybrid Attention Block: RIB window attention + CAB conv branch + i-LN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 32,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        shift: bool = False,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.conv_scale = conv_scale
        self.use_attention = use_attention

        self.norm1 = iLN(dim)
        if use_attention:
            self.attn = RIBWindowAttention(
                dim=dim,
                window_size=window_size,
                num_heads=num_heads,
                rank=rank,
                rib_hidden_dim=rib_hidden_dim,
                rib_n_freqs=rib_n_freqs,
                shift=shift,
            )

        self.conv_block = CAB(
            num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = iLN(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def conv_branch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """spandrel: conv_block on channels-last x, its channel attention as linear
        maps: the convolutions' output and the attention's (B, C) gate, which the
        caller applies with the residual's other factors in one pass."""
        conv_0, activation, conv_1, channel = self.conv_block.cab
        _, squeeze, relu, expand, sigmoid = channel.attention
        x = self.run_conv(
            "cab 1", conv_1, activation(self.run_conv("cab 0", conv_0, x))
        )
        y = F.linear(x.mean((2, 3)), squeeze.weight.flatten(1), squeeze.bias)
        return x, sigmoid(F.linear(relu(y), expand.weight.flatten(1), expand.bias))

    def forward_tokens(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        """spandrel: forward in the token layout (token_layout)."""
        h, w = x_size
        b, n, c = x.shape
        x_norm, std1 = self.norm1(x)
        image = x_norm.view(b, h, w, c)
        conv_x, gate = self.conv_branch(image.permute(0, 3, 1, 2))
        # a view of the convolution's channels-last output: the tokens' layout
        conv_x = conv_x.permute(0, 2, 3, 1)
        scale = std1.view(b, 1, 1, 1)
        # x + std1 * (attention + conv_scale * gate * conv) in one pass per term
        conv_factor = (gate * self.conv_scale)[:, None, None, :] * scale
        if self.use_attention:
            x = torch.addcmul(
                x.view(b, h, w, c), self.attn.forward_tokens(image), scale
            )
            x = x.addcmul_(conv_x, conv_factor)
        else:
            x = torch.addcmul(x.view(b, h, w, c), conv_x, conv_factor)
        del x_norm, image, conv_x
        x = x.view(b, n, c)
        x_norm, std2 = self.norm2(x)
        return x.addcmul_(self.mlp(x_norm), std2.view(b, 1, 1))

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        if token_layout(x, self):
            return self.forward_tokens(x, x_size)
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x, std1 = self.norm1(x)
        x_2d = x.transpose(1, 2).contiguous().view(b, c, h, w)

        # conv branch
        conv_x = self.conv_block(x_2d).view(b, c, -1).transpose(1, 2)
        residual = conv_x * self.conv_scale

        # RIB window attention (channels-first), optional per attention_freq
        if self.use_attention:
            attn_out = self.attn(x_2d)
            attn_x = attn_out.view(b, c, -1).transpose(1, 2)
            residual = residual + self.drop_path(attn_x)

        # i-LN: rescale by std from norm1
        x = shortcut + std1.view(b, 1, 1) * residual

        # FFN with i-LN rescaling
        x_normed, std2 = self.norm2(x)
        ffn_out = self.mlp(x_normed)
        return x + std2.view(b, 1, 1) * self.drop_path(ffn_out)


class AttenBlocksRIB(nn.Module):
    """A stack of HAB_RIB blocks with optional gradient checkpointing."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        mlp_ratio: float,
        drop_path: list[float],
        use_checkpoint: bool = False,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        attention_freq: int = 2,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blocks: list[HAB_RIB] = []
        attn_idx = 0
        for i in range(depth):
            use_attention = i % attention_freq == 0
            blocks.append(
                HAB_RIB(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    rank=rank,
                    rib_hidden_dim=rib_hidden_dim,
                    rib_n_freqs=rib_n_freqs,
                    shift=(attn_idx % 2 == 1) if use_attention else False,
                    use_attention=use_attention,
                )
            )
            if use_attention:
                attn_idx += 1
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        use_chk = self.use_checkpoint and self.training
        for blk in self.blocks:
            if use_chk:
                x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        return x


class RHAG_RIB(Derived):
    """Residual Hybrid Attention Group: AttenBlocksRIB -> 3x3 conv -> residual."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        mlp_ratio: float,
        drop_path: list[float],
        use_checkpoint: bool = False,
        img_size: int = 64,
        patch_size: int = 1,
        resi_connection: str = "1conv",
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        attention_freq: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.residual_group = AttenBlocksRIB(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            use_checkpoint=use_checkpoint,
            rank=rank,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            attention_freq=attention_freq,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv = nn.Identity()
        else:
            raise ValueError(f"unsupported resi_connection: {resi_connection}")

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=dim
        )
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        if isinstance(self.conv, nn.Conv2d) and token_layout(x, self):
            # spandrel: the closing convolution on a channels-last view of the tokens
            y = self.residual_group(x, x_size)
            b, n, c = y.shape
            image = y.view(b, x_size[0], x_size[1], c).permute(0, 3, 1, 2)
            y = self.run_conv("conv", self.conv, image).permute(0, 2, 3, 1)
            return y.reshape(b, n, c).add_(x)
        return (
            self.patch_embed(
                self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
            )
            + x
        )


@store_hyperparameters()
class HEART(Derived):
    """HEART - Hybrid Efficient Attention with Rank-factorized bias Transformer.

    Simplified HAT-iLN: RIB window attention (FlashAttention-friendly) replaces
    the relative position bias table and OCAB is removed. i-LN keeps training
    stable in reduced precision. Pure PyTorch, ONNX-exportable.
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
        num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
        window_size: int = 32,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        upscale: int = 1,
        upsampler: Literal["pixelshuffle"] = "pixelshuffle",
        resi_connection: str = "1conv",
        num_feat: int = 64,
        rank: int = 8,
        rib_hidden_dim: int = 32,
        rib_n_freqs: int = 10,
        attention_freq: int = 2,
    ) -> None:
        super().__init__()

        self.window_size = window_size

        num_in_ch = in_chans
        num_out_ch = in_chans
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            patch_norm=self.patch_norm,
        )
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG_RIB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                rank=rank,
                rib_hidden_dim=rib_hidden_dim,
                rib_n_freqs=rib_n_freqs,
                attention_freq=attention_freq,
            )
            self.layers.append(layer)
        self.norm = AffineTransform(self.num_features)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv_after_body = nn.Identity()
        else:
            raise ValueError(f"unsupported resi_connection: {resi_connection}")

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, iLN)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        can_reflect = (pad_h == 0 or pad_h < h) and (pad_w == 0 or pad_w < w)
        mode = "reflect" if can_reflect else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        if token_layout(x, self):
            # spandrel: the tokens as a channels-last image, without a copy
            return x.view(x.shape[0], *x_size, -1).permute(0, 3, 1, 2)
        return self.patch_unembed(x, x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        x = self.check_image_size(x)

        tokens = token_layout(x, self)
        if tokens:
            # spandrel: channels last from the first convolution to the last, so the
            # trunk's tokens are the image's memory
            x = x.contiguous(memory_format=torch.channels_last)
            x = self.run_conv("conv_first", self.conv_first, x)
            features = self.forward_features(x)
            if isinstance(self.conv_after_body, nn.Conv2d):
                features = self.run_conv(
                    "conv_after_body", self.conv_after_body, features
                )
                x = features.add_(x)
            else:
                x = features + x
        else:
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
        plan = self.tail_plan(x)
        if plan is None:
            x = self.tail(x)
        else:
            trunk_rows = x.shape[2]
            rows = (
                trunk_rows
                if trunk_rows <= WHOLE_TAIL_ROWS
                # an 8x tail's band holds four times a 4x one's pixels per row
                else max(
                    MIN_BAND_ROWS,
                    trunk_rows // (BAND_DIVISOR * max(1, plan[0] // 4) ** 2),
                )
            )
            x = banded_tail(self.tail, x, *plan, rows=rows)

        x = x[:, :, : h * self.upscale, : w * self.upscale]
        return x.contiguous() if tokens else x

    def tail(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_last(self.upsample(self.conv_before_upsample(x)))

    def tail_plan(self, x: torch.Tensor) -> tuple[int, int] | None:
        """(scale, halo) for banding the high-resolution tail in plain eager inference on
        an NVIDIA GPU (any precision); None for the whole pass: tracing, export,
        training, ROCm, the CPU, a tail of no upscaling, or one tail_scale_and_halo
        cannot band."""
        if not (x.is_cuda and torch.version.hip is None and plain_inference(x, self)):
            return None
        if traced():
            return None
        plan = tail_scale_and_halo(
            nn.Sequential(self.conv_before_upsample, self.upsample, self.conv_last)
        )
        return plan if plan is not None and plan[0] > 1 else None
