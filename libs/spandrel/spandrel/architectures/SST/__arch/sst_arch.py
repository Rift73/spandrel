# ruff: noqa
# type: ignore
"""
SST: Windowed Super-Resolution Transformer (consolidated single-file arch)

This file merges the SST reference implementation — originally split across
``sst/archs/arch_utils.py``, ``sst/archs/sst_arch.py`` and
``sst/archs/sst_real_arch.py`` — into one self-contained module, mirroring the
layout of ``drft_arch.py``.

Two model families live here:

1. ``SST`` — a windowed attention transformer for single-image super-resolution.
   Shallow conv projection -> N residual SSTB blocks (each = several windowed
   attention + ConvFFN layers, alternating shifted windows) -> upsampler.
   Selectable positional-bias scheme via ``attn_type``:
       - 'RIB'       : Rank-factored Implicit neural Bias (Flash-compatible,
                       the scheme used by every released SST checkpoint)
       - 'RIBSiren'  : RIB with a SIREN (sinusoidal) coordinate MLP
       - 'FlashBias' : learned low-rank additive bias
       - 'RoPEViT'   : 2D rotary embeddings (mixed / learnable frequencies)
       - 'NoPE'      : no positional bias
       - 'SDPA'/'Naive'/'Flex' : classic relative-position-bias table
   Optional depthwise/linear output gating (``gate_type``) and three upsampler
   heads (``pixelshuffle_direct``, ``pixelshuffle``, ``nn+conv``).

2. ``EDMUNet`` / ``SSTReal`` — a real-world SR *diffusion* model (EDM
   preconditioning) built from the same windowed-attention primitives, with
   timestep/adaLN modulation and a low-res fusion path. Its ``forward`` takes
   ``(x, sigma, lq)`` and returns an image pair, so it is NOT a drop-in for the
   standard SR pipeline — it is included for completeness.

Variants (factory functions, each maps to an ``options/`` YAML config):
    sst_light       -> SSTlight        (dim 48,  5 blocks)
    sst_light_plus  -> SSTlight_Plus   (dim 48,  wide windows)
    sst_base        -> SST             (dim 180, 6 blocks)
    sst_base_plus   -> SST_Plus        (dim 180, wide windows)
    sst_large       -> SSTLarge        (dim 192, 8 blocks)   [aka "SST-L"]
    sst_large_plus  -> SSTLarge_Plus   (dim 192, wide windows)
    sst_xl_plus     -> SSTXL_Plus      (dim 224, 10 blocks)  [aka "SST-XL+"]
    sst_real        -> SSTReal_X4      (EDMUNet diffusion model)

The "_plus" variants only widen ``window_sizes`` from ``[16,32,64,16,32,64]``
to ``[16,32,48,32,48,96]`` (larger receptive field, identical parameter shapes).

Dependencies:
    - torch (recent build; RIB/FlashBias/RoPEViT/NoPE use ``sdpa_kernel``)
    - einops (``rearrange``)
    - flash_attn_interface (OPTIONAL — a source build of FlashAttention-3;
      absent -> automatic fall back to ``F.scaled_dot_product_attention``)

Self-contained: no hard ``basicsr`` import. (spandrel: the traiNNer/basicsr
``ARCH_REGISTRY`` registration at the bottom of the original file is removed; it
would collide with traiNNer's own copy when both are imported.)

NOTE ON CHECKPOINTS: module and parameter names are byte-for-byte identical to
the original three files, so existing SST / EDMUNet ``.safetensors`` /
``.pth`` weights load without renaming.
"""

from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from typing import Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# -----------------------------------------------------------------------------
# Optional runtime dependencies (graceful fallbacks)
# -----------------------------------------------------------------------------

# Flex Attention — requires Triton (Linux). Only needed for attn_type='Flex'.
try:
    from torch.nn.attention.flex_attention import flex_attention

    _FLEX_AVAILABLE = True
except ImportError:
    flex_attention = None
    _FLEX_AVAILABLE = False

# Backend-selection context manager for SDPA. Present on recent PyTorch.
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _SDPA_BACKEND_AVAILABLE = True
except ImportError:  # pragma: no cover - older PyTorch without the context mgr
    import contextlib

    _SDPA_BACKEND_AVAILABLE = False

    class SDPBackend:  # minimal stand-in so attribute refs don't NameError
        FLASH_ATTENTION = None
        CUDNN_ATTENTION = None

    def sdpa_kernel(*args, **kwargs):
        return contextlib.nullcontext()


# FlashAttention-3 (source build). Optional — SDPA is used when unavailable.
try:
    from flash_attn_interface import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


ATTN_TYPE = Literal[
    "NoPE", "Naive", "SDPA", "Flex", "FlashBias", "RIB", "RIBSiren", "RoPEViT"
]
UPSAMPLER_TYPE = Literal["pixelshuffle_direct", "pixelshuffle", "nn+conv"]


# =============================================================================
# Attention helpers
# =============================================================================


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    score = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    score = score + bias
    score = F.softmax(score, dim=-1)
    out = score @ v
    return out


def apply_rpe(table: torch.Tensor, window_size: int):
    def bias_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int):
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]

    return bias_mod


def feat_to_qkv(x: torch.Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x,
        "b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww) c",
        heads=heads,
        wh=window_size[0],
        ww=window_size[1],
        qkv=3,
    )


def feat_to_qkv_fa(x: torch.Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x,
        "b (qkv heads c) (h wh) (w ww) -> qkv (b h w) (wh ww) heads c",
        heads=heads,
        wh=window_size[0],
        ww=window_size[1],
        qkv=3,
    )


def out_to_feat(x, window_size: Sequence[int], h_div: int, w_div: int):
    return rearrange(
        x,
        "(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)",
        h=h_div,
        w=w_div,
        wh=window_size[0],
        ww=window_size[1],
    )


def out_to_feat_fa(x, window_size: Sequence[int], h_div: int, w_div: int):
    return rearrange(
        x,
        "(b h w) (wh ww) heads c -> b (heads c) (h wh) (w ww)",
        h=h_div,
        w=w_div,
        wh=window_size[0],
        ww=window_size[1],
    )


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def init_random_2d_freqs(
    head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True
):
    if head_dim % 4 != 0:
        raise ValueError(
            f"RoPEViT requires head_dim % 4 == 0, but got head_dim={head_dim}"
        )

    freqs_x = []
    freqs_y = []
    mag = 1 / (
        theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim)
    )
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)  # (H, head_dim/2)
    freqs_y = torch.stack(freqs_y, dim=0)  # (H, head_dim/2)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)  # (2, H, head_dim/2)
    return freqs


def compute_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor):
    """
    freqs: (2, H, head_dim/2)
    t_x, t_y: (N,)
    return: (H, N, head_dim/2) complex
    """
    device_type = freqs.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        freqs_x = t_x.float().unsqueeze(-1) @ freqs[0].float().unsqueeze(
            -2
        )  # (H, N, D/2)
        freqs_y = t_y.float().unsqueeze(-1) @ freqs[1].float().unsqueeze(
            -2
        )  # (H, N, D/2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)  # complex
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError(f"Unexpected shape: freqs_cis={freqs_cis.shape}, x={x.shape}")
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


# =============================================================================
# Core modules: Windowed Attention + LayerNorm
# =============================================================================


class WindowAttention2D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_type: ATTN_TYPE = "Flex",
        rank: Optional[int] = None,
        attn_func=None,
        shift: bool = False,
        rib_hidden_dim: Optional[int] = None,
        rib_n_freqs: Optional[int] = None,
        gate_type=None,
    ):
        super().__init__()
        self.dim = dim
        window_size = (
            (window_size, window_size) if isinstance(window_size, int) else window_size
        )
        self.window_size = window_size
        self.num_heads = num_heads

        assert (
            dim % num_heads == 0
        ), f"Embedding dimension {dim} should be divisible by number of heads {num_heads}."

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

        self.attn_type = attn_type
        if attn_func is None:
            if attn_type == "Flex":
                raise ValueError(
                    "For Flex Attention, compiled attn_func must be provided."
                )
            if FLASH_ATTN_AVAILABLE and attn_type in (
                "FlashBias",
                "RIB",
                "RIBSiren",
                "RoPEViT",
            ):
                self.attn_func = partial(
                    flash_attn_func, softmax_scale=1.0, causal=False
                )
            elif not FLASH_ATTN_AVAILABLE and attn_type in (
                "FlashBias",
                "RIB",
                "RIBSiren",
                "RoPEViT",
            ):
                self.attn_func = partial(
                    F.scaled_dot_product_attention,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=1.0,
                )
            else:
                self.attn_func = partial(
                    F.scaled_dot_product_attention, dropout_p=0.0, is_causal=False
                )
        else:
            self.attn_func = attn_func

        self.qkv_func = (
            feat_to_qkv_fa
            if (
                FLASH_ATTN_AVAILABLE
                and attn_type in ("FlashBias", "RIB", "RIBSiren", "RoPEViT", "NoPE")
            )
            else feat_to_qkv
        )
        self.out_func = (
            out_to_feat_fa
            if (
                FLASH_ATTN_AVAILABLE
                and attn_type in ("FlashBias", "RIB", "RIBSiren", "RoPEViT", "NoPE")
            )
            else out_to_feat
        )

        if attn_type not in ("FlashBias", "RIB", "RIBSiren", "RoPEViT", "NoPE"):
            self.relative_position_bias = nn.Parameter(
                torch.randn(
                    num_heads, (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
                ).to(torch.float32)
                * 0.001
            )
            if self.attn_type == "Flex":
                self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
            else:  # Naive or SDPA
                self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)

        head_dim = dim // num_heads
        self.rank = 256 - head_dim if rank is None else rank

        if self.attn_type == "FlashBias":
            self.flashbias_q = nn.Parameter(
                torch.zeros(num_heads, window_size[0] * window_size[1], self.rank)
            )
            self.flashbias_k = nn.Parameter(
                torch.zeros(num_heads, window_size[0] * window_size[1], self.rank)
            )

        if self.attn_type in ("RIB", "RIBSiren"):
            Wh, Ww = window_size
            yy, xx = torch.meshgrid(torch.arange(Wh), torch.arange(Ww), indexing="ij")
            coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()  # (N,2)
            if Ww > 0:
                coords[:, 0] = (2.0 * (coords[:, 0] + 0.5) / Ww) - 1.0
            else:
                coords[:, 0] = 0
            if Wh > 0:
                coords[:, 1] = (2.0 * (coords[:, 1] + 0.5) / Wh) - 1.0
            else:
                coords[:, 1] = 0

            self.n_freqs = (
                0
                if self.attn_type == "RIBSiren"
                else 10
                if rib_n_freqs is None
                else rib_n_freqs
            )
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
            self.register_buffer("rib_coords", coords, persistent=False)

            n_input = 2 + 4 * self.n_freqs
            hidden_d = 32 if rib_hidden_dim is None else rib_hidden_dim

            self.to_hidden = nn.Parameter(torch.empty(n_input, hidden_d))
            self.hidden_b = nn.Parameter(torch.zeros(1, hidden_d))
            self.to_q = nn.Parameter(torch.empty(num_heads, hidden_d, self.rank))
            self.to_k = nn.Parameter(torch.empty(num_heads, hidden_d, self.rank))

            if self.attn_type == "RIBSiren":
                self.rib_omega0 = 30.0

            self._reset_rib_parameters(
                siren=(self.attn_type == "RIBSiren"),
                omega0=getattr(self, "rib_omega0", 1.0),
            )

        if self.attn_type == "RoPEViT":
            if head_dim % 4 != 0:
                head_dim += 4 - (head_dim % 4)

            Wh, Ww = window_size
            t_x, t_y = init_t_xy(end_x=Ww, end_y=Wh)
            self.register_buffer("rope_t_x", t_x, persistent=False)
            self.register_buffer("rope_t_y", t_y, persistent=False)

            rope_mixed = True  # Learnable frequencies
            rope_use_rpb = False  # Our goal is not to use RPB for Flash Attention
            rope_theta = 10.0  # recommended value
            self.rope_mixed = rope_mixed
            freqs = init_random_2d_freqs(
                head_dim=head_dim,
                num_heads=self.num_heads,
                theta=rope_theta,
                rotate=self.rope_mixed,
            )
            if self.rope_mixed:
                self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
            else:
                self.register_buffer("rope_freqs", freqs, persistent=False)
                freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
                self.register_buffer("rope_freqs_cis", freqs_cis, persistent=False)
            self.rope_use_rpb = rope_use_rpb

        self.shift = shift

        self.gate_type = gate_type
        if gate_type is not None:
            if gate_type == "Linear":
                self.gate = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0), nn.Sigmoid())
            elif gate_type == "DWC":
                self.gate = nn.Sequential(
                    nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Sigmoid(),
                )
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")

    def _reset_rib_parameters(self, siren: bool, omega0: float = 30.0):
        n_in = self.to_hidden.shape[0]

        if siren:
            bound = 1.0 / n_in
            nn.init.uniform_(self.to_hidden, -bound, bound)

            n_in2 = self.to_q.shape[1]  # hidden_d
            bound2 = math.sqrt(6.0 / n_in2) / omega0
            nn.init.uniform_(self.to_q, -bound2, bound2)
            nn.init.uniform_(self.to_k, -bound2, bound2)
        else:
            nn.init.normal_(self.to_hidden, mean=0.0, std=0.05)
            nn.init.normal_(self.to_q, mean=0.0, std=0.05)
            nn.init.normal_(self.to_k, mean=0.0, std=0.05)

    def _flash_cat_attn(self, q, k, v, pos_q, pos_k):
        if FLASH_ATTN_AVAILABLE:
            Bwin, Nq, H, D = q.shape
            Nkv = k.shape[1]
            pos_q = pos_q.transpose(1, 2)  # Bwin, N, heads, R
            pos_k = pos_k.transpose(1, 2)
        else:
            Bwin, H, Nq, D = q.shape
            Nkv = k.shape[2]
        R = pos_q.shape[-1]

        q = q * (D**-0.5)
        pos_q = pos_q * (R**-0.5)
        q_cat = torch.cat([q.to(torch.bfloat16), pos_q.to(torch.bfloat16)], dim=-1)
        k_cat = torch.cat([k.to(torch.bfloat16), pos_k.to(torch.bfloat16)], dim=-1)
        if FLASH_ATTN_AVAILABLE:
            v_cat = torch.cat(
                [
                    v.to(torch.bfloat16),
                    torch.zeros(
                        (Bwin, Nkv, H, R), device=v.device, dtype=torch.bfloat16
                    ),
                ],
                dim=-1,
            )
        else:
            v_cat = torch.cat(
                [
                    v.to(torch.bfloat16),
                    torch.zeros(
                        (Bwin, H, Nkv, R), device=v.device, dtype=torch.bfloat16
                    ),
                ],
                dim=-1,
            )

        d_total = q_cat.shape[-1]
        pad = (8 - (d_total % 8)) % 8
        if pad:
            q_cat = F.pad(q_cat, (0, pad), mode="constant", value=0)
            k_cat = F.pad(k_cat, (0, pad), mode="constant", value=0)
            v_cat = F.pad(v_cat, (0, pad), mode="constant", value=0)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            out = self.attn_func(
                q_cat.contiguous(),
                k_cat.contiguous(),
                v_cat.contiguous(),
            )[:, :, :, :D]
            # out = (
            #     F.softmax(q_cat @ k_cat.transpose(-2, -1), dim=-1) @ v_cat
            # )[:, :, :, :D]
        return out

    def _ropevit_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if FLASH_ATTN_AVAILABLE:
            q = q.transpose(1, 2)  # Bwin, heads, N, head_dim
            k = k.transpose(1, 2)

        headdim = q.shape[-1]
        if headdim % 4 != 0:
            pad = 4 - (headdim % 4)
            q = F.pad(q, (0, pad), mode="constant", value=0)
            k = F.pad(k, (0, pad), mode="constant", value=0)

        if self.rope_mixed:
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
        else:
            freqs_cis = self.rope_freqs_cis.to(device=q.device)

        q, k = apply_rotary_emb(q, k, freqs_cis)
        if headdim % 4 != 0:
            q = q[..., :headdim]
            k = k[..., :headdim]

        if FLASH_ATTN_AVAILABLE:
            q = q.transpose(1, 2)  # Bwin, N, heads, head_dim
            k = k.transpose(1, 2)

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            out = self.attn_func(
                q.to(torch.bfloat16).contiguous(),
                k.to(torch.bfloat16).contiguous(),
                v.to(torch.bfloat16).contiguous(),
            )
        return out

    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
        idxs_window = []
        for head in range(heads):
            for h in range(window_size**2):
                for w in range(window_size**2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        return torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)

    def pad_to_win(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def roll(self, x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
        if not self.shift:
            return x
        shift_size = (window_size[0] // 2, window_size[1] // 2)
        return torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))

    def unroll(self, x: torch.Tensor, window_size: Sequence[int]) -> torch.Tensor:
        if not self.shift:
            return x
        shift_size = (window_size[0] // 2, window_size[1] // 2)
        return torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        if self.gate_type is not None:
            gate = self.gate(x)
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        x = self.roll(x, self.window_size)
        h_div, w_div = (
            x.shape[2] // self.window_size[0],
            x.shape[3] // self.window_size[1],
        )

        qkv = self.to_qkv(x)
        dtype = qkv.dtype
        qkv = self.qkv_func(qkv, self.window_size, self.num_heads)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # (B*nwin, heads, N, head_dim) or (B*nwin, N, heads, head_dim)

        if self.attn_type == "Flex":
            head_dim = q.shape[-1]
            target_dim = 1 << (head_dim - 1).bit_length()
            if head_dim != target_dim:
                if target_dim < head_dim:
                    target_dim = target_dim << 1
                q = F.pad(q, (0, target_dim - head_dim), mode="constant", value=0)
                k = F.pad(k, (0, target_dim - head_dim), mode="constant", value=0)
                v = F.pad(v, (0, target_dim - head_dim), mode="constant", value=0)
            q = q * (head_dim**-0.5)
            out = self.attn_func(q, k, v, score_mod=self.get_rpe, scale=1.0)[
                :, :, :, :head_dim
            ]

        elif self.attn_type == "SDPA":
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
            out = self.attn_func(q, k, v, attn_mask=bias)

        elif self.attn_type == "Naive":
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
            out = attention(q, k, v, bias)

        elif self.attn_type == "FlashBias":
            Bwin = q.shape[0]

            q_bias = (
                self.flashbias_q.to(dtype=q.dtype, device=q.device)
                .unsqueeze(0)
                .expand(Bwin, -1, -1, -1)
            )
            k_bias = (
                self.flashbias_k.to(dtype=k.dtype, device=k.device)
                .unsqueeze(0)
                .expand(Bwin, -1, -1, -1)
            )

            out = self._flash_cat_attn(q, k, v, q_bias, k_bias)

        elif self.attn_type == "RIB":
            Bwin = q.shape[0]

            coords = self.rib_coords.to(dtype=q.dtype, device=q.device)
            hidden_w = self.to_hidden.to(dtype=q.dtype, device=q.device)
            hidden_b = self.hidden_b.to(dtype=q.dtype, device=q.device)
            intermediate = F.relu(
                coords @ hidden_w + hidden_b  # N, hidden_d
            )
            q_pos = (
                torch.einsum(
                    "nd,hdr->hnr",
                    intermediate,
                    self.to_q.to(dtype=q.dtype, device=q.device),
                )
                .unsqueeze(0)
                .expand(Bwin, -1, -1, -1)
            )  # Bwin, H, Nq, R
            k_pos = (
                torch.einsum(
                    "nd,hdr->hnr",
                    intermediate,
                    self.to_k.to(dtype=k.dtype, device=k.device),
                )
                .unsqueeze(0)
                .expand(Bwin, -1, -1, -1)
            )  # Bwin, H, Nkv, R
            out = self._flash_cat_attn(q, k, v, q_pos, k_pos)

        elif self.attn_type == "RIBSiren":
            Bwin = q.shape[0]

            coords = self.rib_coords.to(device=q.device, dtype=torch.float32)
            hidden_w = self.to_hidden.to(device=q.device, dtype=torch.float32)

            pre = coords @ hidden_w + self.hidden_b.to(
                device=q.device, dtype=torch.float32
            )
            intermediate = torch.sin(self.rib_omega0 * pre)
            to_q = self.to_q.to(device=q.device, dtype=torch.float32)
            to_k = self.to_k.to(device=q.device, dtype=torch.float32)

            q_pos = (
                torch.einsum("nd,hdr->hnr", intermediate, to_q)
                .to(dtype=q.dtype)
                .unsqueeze(0)
                .expand(Bwin, -1, -1, -1)
            )
            k_pos = (
                torch.einsum("nd,hdr->hnr", intermediate, to_k)
                .to(dtype=k.dtype)
                .unsqueeze(0)
                .expand(Bwin, -1, -1, -1)
            )

            out = self._flash_cat_attn(q, k, v, q_pos, k_pos)

        elif self.attn_type == "RoPEViT":
            out = self._ropevit_attn(q, k, v)

        elif self.attn_type == "NoPE":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                out = self.attn_func(
                    q.to(torch.bfloat16).contiguous(),
                    k.to(torch.bfloat16).contiguous(),
                    v.to(torch.bfloat16).contiguous(),
                ).to(dtype)

        else:
            raise NotImplementedError(
                f"Attention type {self.attn_type} is not supported."
            )

        out = self.out_func(out, self.window_size, h_div, w_div)
        out = self.unroll(out, self.window_size).to(dtype)[:, :, :h, :w]
        if self.gate_type is not None:
            out = out * gate
        out = self.to_out(out)
        return out

    def extra_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, attn_type={self.attn_type}, shift={self.shift}, USE_FLASH_ATTN_SOURCE_BUILD={FLASH_ATTN_AVAILABLE}"


class LayerNorm(nn.Module):
    def __init__(
        self, normalized_shape, eps=1e-6, data_format="channels_first", use_affine=True
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if use_affine else None
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if use_affine else None
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            if self.training:
                return (
                    F.layer_norm(
                        x.transpose(1, -1).contiguous(),
                        self.normalized_shape,
                        self.weight,
                        self.bias,
                        self.eps,
                    )
                    .transpose(1, -1)
                    .contiguous()
                )
            else:
                return F.layer_norm(
                    x.transpose(1, -1),
                    self.normalized_shape,
                    self.weight,
                    self.bias,
                    self.eps,
                ).transpose(1, -1)


# =============================================================================
# Upsampling
# =============================================================================


class Upsampler(nn.Module):
    """
    X4/PixelShuffle is different from the common implementations for weight interpolation initialization:
        - Common: Convx2 -> PSx2 -> Act -> Convx2 -> PSx2 -> Act -> Conv
        - Ours: Convx4 -> PSx4 -> Act -> Conv
    """

    def __init__(
        self,
        dim,
        upscaling_factor,
        upsampler_type: UPSAMPLER_TYPE,
        intermediate_dim: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.upscaling_factor = upscaling_factor
        self.upsampler_type = upsampler_type

        self.target_weight_name = "up_conv.weight"
        self.target_bias_name = "up_conv.bias"

        if upsampler_type == "pixelshuffle_direct":
            self.up_conv = nn.Conv2d(
                dim, 3 * (upscaling_factor**2), kernel_size=3, padding=1
            )

        elif upsampler_type == "pixelshuffle":
            num_feat = intermediate_dim
            self.feature_conv = nn.Conv2d(dim, num_feat, kernel_size=3, padding=1)
            self.up_conv = nn.Conv2d(
                num_feat, num_feat * (upscaling_factor**2), kernel_size=3, padding=1
            )
            self.final_conv = nn.Conv2d(num_feat, 3, kernel_size=3, padding=1)

        elif upsampler_type == "nn+conv":
            num_feat = intermediate_dim
            f = []
            f.extend(
                [
                    nn.Conv2d(dim, num_feat, kernel_size=3, padding=1),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ]
            )
            if (upscaling_factor & (upscaling_factor - 1)) == 0:
                for _ in range(int(math.log2(upscaling_factor))):
                    f.extend(
                        [
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
                            nn.LeakyReLU(negative_slope=0.1, inplace=True),
                        ]
                    )
            elif upscaling_factor == 3:
                f.extend(
                    [
                        nn.Upsample(scale_factor=3, mode="nearest"),
                        nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    ]
                )
            else:
                raise ValueError(
                    f"upscaling_factor {upscaling_factor} is not supported. Supported factors: 2^n and 3."
                )
            f.append(nn.Conv2d(num_feat, 3, kernel_size=3, padding=1))
            self.f = nn.Sequential(*f)
            self.f_img = nn.Sequential(
                nn.Conv2d(3, dim, 1),
                nn.Conv2d(dim, dim, kernel_size=7, padding=3),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(dim, dim, 1),
            )
        else:
            raise ValueError(f"upsampler_type {upsampler_type} is not supported.")

    def extra_repr(self) -> str:
        return f"upscaling_factor={self.upscaling_factor}, upsampler_type={self.upsampler_type}"

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.upsampler_type == "pixelshuffle_direct":
            x = self.up_conv(x)
            if skip is not None:
                x = x + torch.repeat_interleave(
                    skip, repeats=self.upscaling_factor**2, dim=1
                )
            x = F.pixel_shuffle(x, self.upscaling_factor)
        elif self.upsampler_type == "pixelshuffle":
            x = F.leaky_relu(self.feature_conv(x), negative_slope=0.1, inplace=True)
            x = self.up_conv(x)
            x = F.pixel_shuffle(x, self.upscaling_factor)
            x = self.final_conv(x)
            if skip is not None:
                x = x + F.interpolate(
                    skip, scale_factor=self.upscaling_factor, mode="nearest"
                )
        elif self.upsampler_type == "nn+conv":
            x = self.f(x + self.f_img(skip)) if skip is not None else self.f(x)
        else:
            raise ValueError(f"upsampler_type {self.upsampler_type} is not supported.")
        return x


class ImageArchitecture(nn.Module):
    """
    For ease of upsampler management
    """

    def build_upsampler(
        self,
        dim,
        upscaling_factor,
        upsampler_type: UPSAMPLER_TYPE,
        intermediate_dim: int = 64,
    ):
        self.upsampler = Upsampler(
            dim, upscaling_factor, upsampler_type, intermediate_dim=intermediate_dim
        )

    def interpolate_upsampler(self, state_dict):
        if self.upsampler.upsampler_type != "nn+conv":
            sd = deepcopy(state_dict)

            target_weight_name = f"upsampler.{self.upsampler.target_weight_name}"
            target_bias_name = f"upsampler.{self.upsampler.target_bias_name}"
            target_weight = sd[target_weight_name]
            target_bias = sd[target_bias_name]

            oc = target_weight.shape[0]
            ic = target_weight.shape[1]

            if self.upsampler.upsampler_type == "pixelshuffle_direct":
                r2 = oc / 3
            elif self.upsampler.upsampler_type == "pixelshuffle":
                r2 = oc / ic
            else:
                raise ValueError

            sd_scale = int(round(math.sqrt(r2)))
            cur_scale = self.upsampler.upscaling_factor

            if sd_scale != cur_scale:
                _log_info(
                    f"Interpolating Upsampler from x{sd_scale} to x{cur_scale}..."
                )

                out_dim = target_bias.shape[0] // (sd_scale**2)  # 3 or feat_dim

                def interpolate_kernel(kernel, scale_in, scale_out, out_dim):
                    _, _, kh, kw = kernel.shape
                    kernel = rearrange(
                        kernel,
                        "(dim rh rw) cin kh kw -> (cin kh kw) dim rh rw",
                        dim=out_dim,
                        rh=scale_in,
                        rw=scale_in,
                    )
                    kernel = F.interpolate(
                        kernel,
                        size=(scale_out, scale_out),
                        mode="bilinear",
                        align_corners=False,
                    )
                    kernel = rearrange(
                        kernel,
                        "(cin kh kw) dim rh rw -> (dim rh rw) cin kh kw",
                        kh=kh,
                        kw=kw,
                    )
                    return kernel

                def interpolate_bias(bias, scale_in, scale_out, out_dim):
                    bias = rearrange(
                        bias,
                        "(dim rh rw) -> 1 dim rh rw",
                        dim=out_dim,
                        rh=scale_in,
                        rw=scale_in,
                    )
                    bias = F.interpolate(
                        bias,
                        size=(scale_out, scale_out),
                        mode="bilinear",
                        align_corners=False,
                    )
                    bias = rearrange(bias, "1 dim rh rw -> (dim rh rw)")
                    return bias

                sd[target_weight_name] = interpolate_kernel(
                    target_weight, sd_scale, cur_scale, out_dim
                )
                sd[target_bias_name] = interpolate_bias(
                    target_bias, sd_scale, cur_scale, out_dim
                )

                return sd

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Back-compat: checkpoints saved before the _window_sizes metadata buffer
        # was introduced won't contain it. Inject the value this model was built
        # with so strict loading still succeeds (the buffer is metadata only and
        # does not affect the loaded weights).
        buf = getattr(self, "_window_sizes", None)
        if buf is not None and "_window_sizes" not in state_dict:
            state_dict = {**state_dict, "_window_sizes": buf}
        state_dict = self.interpolate_upsampler(state_dict)
        super().load_state_dict(state_dict, strict)


def _log_info(msg: str) -> None:
    """Log an info message via the traiNNer/basicsr root logger if present, else stdlib logging."""
    for module_name in ("traiNNer.utils", "basicsr.utils"):
        try:
            get_root_logger = __import__(
                module_name, fromlist=["get_root_logger"]
            ).get_root_logger
            get_root_logger().info(msg)
            return
        except Exception:
            continue
    import logging

    logging.getLogger("sst_arch").info(msg)


# =============================================================================
# SST: Super-Resolution Transformer
# =============================================================================


class ConvFFN(nn.Module):
    def __init__(self, dim: int, exp_ratio: float | int, kernel_size: int = 3):
        super().__init__()
        d_in = dim
        d_hidden = int(dim * exp_ratio)

        self.proj = nn.Conv2d(d_in, d_hidden, kernel_size=1, stride=1, padding=0)
        self.dwc = nn.Conv2d(
            d_hidden,
            d_hidden,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=d_hidden,
        )
        self.agg = nn.Conv2d(d_hidden, d_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(x))
        x = x + F.gelu(self.dwc(x))
        x = self.agg(x)
        return x


class STL(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_head: int,
        attn_type: int,
        shift: bool,
        rank: Optional[int] = None,
        rib_hidden_dim: Optional[int] = None,
        rib_n_freqs: Optional[int] = None,
        exp_ratio: float | int = 2,
        attn_func: Optional[any] = None,
        gate_type: Optional[str] = None,
    ):
        super().__init__()
        if attn_type == "CPE":
            attn_type = "NoPE"
            self.use_cpe = True
            self.cpe = nn.Conv2d(
                dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
            )
        else:
            self.use_cpe = False

        self.norm_attn = LayerNorm(dim)
        self.attn = WindowAttention2D(
            dim=dim,
            window_size=window_size,
            num_heads=num_head,
            attn_type=attn_type,
            shift=shift,
            rank=rank,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            attn_func=attn_func,
            gate_type=gate_type,
        )

        self.norm_ffn = LayerNorm(dim)
        self.ffn = ConvFFN(dim, exp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cpe:
            x = x + self.cpe(x)
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SSTB(nn.Sequential):
    def __init__(
        self,
        dim: int,
        window_sizes: Sequence[int],
        num_heads: Sequence[int],
        attn_type: str,
        ranks: Sequence[Optional[int]],
        rib_hidden_dim: Optional[int] = None,
        rib_n_freqs: Optional[int] = None,
        exp_ratio: float | int = 2,
        attn_func: Optional[any] = None,
        gate_type: Optional[str] = None,
    ):
        use_shift = all([ws == window_sizes[0] for ws in window_sizes])
        super().__init__(
            *[
                STL(
                    dim=dim,
                    window_size=ws,
                    num_head=nh,
                    attn_type=attn_type,
                    shift=(i % 2 == 1) and use_shift,
                    rank=rank,
                    rib_hidden_dim=rib_hidden_dim,
                    rib_n_freqs=rib_n_freqs,
                    exp_ratio=exp_ratio,
                    attn_func=attn_func,
                    gate_type=gate_type,
                )
                for i, (ws, nh, rank) in enumerate(zip(window_sizes, num_heads, ranks))
            ]
            + [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)


class SST(ImageArchitecture):
    def __init__(
        self,
        dim: int,
        window_sizes: Sequence[int],
        num_heads: Sequence[int],
        n_blocks: int,
        exp_ratio: float | int = 2,
        attn_type: ATTN_TYPE = "RIB",
        rib_hidden_dim: Optional[int] = 64,
        rib_n_freqs: Optional[int] = 10,
        upscaling_factor: int = 2,
        upsampler_type: UPSAMPLER_TYPE = "pixelshuffle_direct",
        ranks: Sequence[Optional[int]] = None,
        gate_type: Optional[str] = None,
        intermediate_dim: int = 64,
    ):
        super().__init__()
        assert len(window_sizes) == len(
            num_heads
        ), "window_sizes and num_heads must have the same length."
        if attn_type in ["RIB", "RIBSiren"]:
            assert len(window_sizes) == len(
                ranks
            ), "window_sizes and ranks must have the same length."
        else:
            if ranks is None:
                ranks = [None] * len(window_sizes)

        if attn_type == "Flex":
            if not _FLEX_AVAILABLE:
                raise RuntimeError(
                    "attn_type='Flex' requires torch.nn.attention.flex_attention "
                    "(Triton, Linux). It is unavailable in this environment."
                )
            attn_func = torch.compile(flex_attention, dynamic=True)
        else:
            attn_func = None
        self.proj = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.body = nn.Sequential(
            *[
                SSTB(
                    dim=dim,
                    window_sizes=window_sizes,
                    num_heads=num_heads,
                    attn_type=attn_type,
                    ranks=ranks,
                    rib_hidden_dim=rib_hidden_dim,
                    rib_n_freqs=rib_n_freqs,
                    exp_ratio=exp_ratio,
                    attn_func=attn_func,
                    gate_type=gate_type,
                )
                for _ in range(n_blocks)
            ]
            + [LayerNorm(dim), nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)]
        )

        self.build_upsampler(
            dim, upscaling_factor, upsampler_type, intermediate_dim=intermediate_dim
        )
        self.init_weight()

        # Self-describing metadata: RIB/RoPEViT/NoPE attention does NOT encode
        # window_sizes in any recoverable weight, so record it as a persistent
        # buffer. This lets loaders (e.g. spandrel) reconstruct the exact model
        # from the checkpoint alone. Back-compat with pre-buffer checkpoints is
        # handled in ImageArchitecture.load_state_dict (it injects this buffer
        # when the incoming state dict lacks it).
        self.register_buffer(
            "_window_sizes",
            torch.tensor(list(window_sizes), dtype=torch.long),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        feat = feat + self.body(feat)
        x = self.upsampler(feat, x)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.Conv2d):
                if m.weight.shape[-1] == 1:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    pass  # Use standard initialization for 3x3 convs
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# =============================================================================
# SSTReal / EDMUNet: real-world SR diffusion model (EDM preconditioning)
# =============================================================================


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(2).unsqueeze(3)) + shift.unsqueeze(2).unsqueeze(3)


class ConvSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Sequential(
            nn.Conv2d(dim, 2 * hidden_dim, kernel_size=1, bias=bias),
            nn.Conv2d(
                2 * hidden_dim,
                2 * hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=2 * hidden_dim,
            ),
        )
        self.w3 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class RealSTL(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_head: int,
        attn_type: int,
        shift: bool,
        rank: Optional[int] = None,
        rib_hidden_dim: Optional[int] = None,
        rib_n_freqs: Optional[int] = None,
        exp_ratio: float | int = 2,
        attn_func: Optional[any] = None,
        gate_type: Optional[str] = None,
        drop_p=0.0,
    ):
        super().__init__()

        self.norm_attn = LayerNorm(dim, use_affine=False)
        self.attn = WindowAttention2D(
            dim=dim,
            window_size=window_size,
            num_heads=num_head,
            attn_type=attn_type,
            shift=shift,
            rank=rank,
            rib_hidden_dim=rib_hidden_dim,
            rib_n_freqs=rib_n_freqs,
            attn_func=attn_func,
            gate_type=gate_type,
        )

        self.norm_ffn = LayerNorm(dim, use_affine=False)
        self.ffn = ConvSwiGLUFFN(dim, int(dim * exp_ratio), drop=drop_p)

    def forward(
        self,
        x: torch.Tensor,
        attn_shift,
        attn_scale,
        attn_gate,
        ffn_shift,
        ffn_scale,
        ffn_gate,
    ) -> torch.Tensor:
        x = x + self.attn(
            modulate(self.norm_attn(x), attn_shift, attn_scale)
        ) * attn_gate.unsqueeze(2).unsqueeze(3)
        x = x + self.ffn(
            modulate(self.norm_ffn(x), ffn_shift, ffn_scale)
        ) * ffn_gate.unsqueeze(2).unsqueeze(3)
        return x


class NNx4Upsampler(nn.Sequential):
    def __init__(self, dim_in, dim_up):
        super().__init__(
            nn.Conv2d(dim_in, dim_up, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim_up, dim_up, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim_up, dim_up, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim_up, 3, kernel_size=3, stride=1, padding=1),
        )


class RealSSTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        window_sizes: Sequence[int],
        num_heads: Sequence[int],
        attn_type: str,
        ranks: Sequence[Optional[int]],
        rib_hidden_dim: Optional[int] = None,
        rib_n_freqs: Optional[int] = None,
        exp_ratio: int | float = 2,
        attn_func: Optional[any] = None,
        gate_type: Optional[str] = None,
        fuse_lr: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not (len(window_sizes) == len(num_heads) == len(ranks)):
            raise ValueError(
                "window_sizes, num_heads, and ranks must have identical lengths: "
                f"{len(window_sizes)}, {len(num_heads)}, {len(ranks)}"
            )

        if fuse_lr:
            self.fuse_lr = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)

        use_shift = all(window_size == window_sizes[0] for window_size in window_sizes)
        self.blocks = nn.ModuleList(
            [
                RealSTL(
                    dim=in_channels,
                    window_size=window_size,
                    num_head=num_head,
                    attn_type=attn_type,
                    shift=(block_idx % 2 == 1) and use_shift,
                    rank=rank,
                    rib_hidden_dim=rib_hidden_dim,
                    rib_n_freqs=rib_n_freqs,
                    exp_ratio=exp_ratio,
                    attn_func=attn_func,
                    gate_type=gate_type,
                    drop_p=kwargs.get("drop_p", 0.0),
                )
                for block_idx, (window_size, num_head, rank) in enumerate(
                    zip(window_sizes, num_heads, ranks)
                )
            ]
        )
        self.out_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        self.adaLN_modulation = nn.Linear(emb_channels, 6 * in_channels, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)
        self.gamma = nn.Parameter(torch.full((1, in_channels, 1, 1), 1e-3))

    def forward(
        self, x: torch.Tensor, lr=None, embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x

        attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate = (
            self.adaLN_modulation(embed).chunk(6, dim=-1)
        )

        if lr is not None:
            x = self.fuse_lr(torch.cat([x, lr], dim=1))

        for block in self.blocks:
            x = block(
                x, attn_shift, attn_scale, attn_gate, ffn_shift, ffn_scale, ffn_gate
            )
        x = self.out_conv(x) * self.gamma
        return residual + x


class SSTReal(nn.Module):
    def __init__(self, **model_kwargs):
        super().__init__()
        dim = model_kwargs["dim"]
        intermediate_dim = model_kwargs["intermediate_dim"]
        emb_channels = model_kwargs["emb_channels"]
        noise_channels = model_kwargs["noise_channels"]

        self.fuse_lr_till = model_kwargs["fuse_lr_till"]
        self.ig_idx = model_kwargs["ig_idx"]

        self.map_noise = TimestepEmbedder(
            hidden_size=emb_channels, frequency_embedding_size=noise_channels
        )

        self.proj_in = nn.Conv2d(3 * 4 * 4, dim, kernel_size=3, stride=1, padding=1)
        self.proj_lr = nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1)
        self.proj_body = nn.ModuleList(
            [
                RealSSTBlock(
                    in_channels=dim, fuse_lr=i <= self.fuse_lr_till, **model_kwargs
                )
                for i in range(model_kwargs["num_blocks"])
            ]
        )

        self.up_i = NNx4Upsampler(dim, intermediate_dim)
        self.up_f = NNx4Upsampler(dim, intermediate_dim)

    def no_weight_decay(self):
        no_decay = set()

        for module_name, module in self.named_modules():
            if isinstance(module, NNx4Upsampler):
                for param_name, _ in module.named_parameters(
                    prefix=module_name, recurse=True
                ):
                    if param_name.endswith(".bias"):
                        no_decay.add(param_name)

                for child_name, child in reversed(list(module.named_children())):
                    if isinstance(child, nn.Conv2d) and child.out_channels == 3:
                        no_decay.add(f"{module_name}.{child_name}.weight")
                        if child.bias is not None:
                            no_decay.add(f"{module_name}.{child_name}.bias")
                        break

            if isinstance(module, WindowAttention2D):
                for param_name in (
                    "to_hidden",
                    "hidden_b",
                    "to_q",
                    "to_k",
                ):  # RIB-related parameters
                    if hasattr(module, param_name):
                        no_decay.add(f"{module_name}.{param_name}")

        for param_name, _ in self.named_parameters():
            if (
                param_name == "proj_lr.bias"
                or param_name.endswith(".adaLN_modulation.bias")
                or param_name.endswith(".fuse_lr.bias")
            ):
                no_decay.add(param_name)

        return no_decay

    def forward(self, x, noise, lq, **kwargs):
        noise_emb = self.map_noise(noise)
        x = self.proj_in(F.pixel_unshuffle(x, downscale_factor=4))
        feat_skip = x
        lr_emb = self.proj_lr(lq)
        for idx, block in enumerate(self.proj_body):
            if idx <= self.fuse_lr_till:
                x = block(x, lr=lr_emb, embed=noise_emb)
            else:
                x = block(x, embed=noise_emb)
            if idx == self.ig_idx:
                x_i = self.up_i(x + feat_skip)
        x_f = self.up_f(x + feat_skip)
        return x_i, x_f


class EDMUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        up_list=None,
        down_list=None,
        use_skip=True,
        sampling_ig_lambda=1.0,
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.up_list = up_list
        self.down_list = down_list
        self.use_skip = use_skip
        self.sampling_ig_lambda = sampling_ig_lambda
        self.model = SSTReal(**model_kwargs)

    def forward(self, x, sigma, lq, lq_up=None, force_fp32=False, **model_kwargs):
        return_pair = model_kwargs.get("return_pair", False)

        x = x.to(torch.float32)
        x_lr = lq
        sigma = sigma.reshape(-1, 1, 1, 1).to(torch.float32)
        model_dtype = torch.float32
        if self.use_skip:
            c_skip = self.sigma_data**2 / ((sigma / 0.1) ** 2 + self.sigma_data**2)
            c_out = (sigma / 0.1) / ((sigma / 0.1) ** 2 + self.sigma_data**2).sqrt()
        else:
            c_skip = self.sigma_data**2 / ((sigma / 0.001) ** 2 + self.sigma_data**2)
            c_out = (sigma / 0.001) / ((sigma / 0.001) ** 2 + self.sigma_data**2).sqrt()
        c_noise = ((sigma + 0.002).log() / 4).to(model_dtype)
        c_skip = c_skip.to(torch.float32)
        c_out = c_out.to(torch.float32)

        model_input = x.to(model_dtype)

        x_i, x_f = self.model(model_input, c_noise.flatten(), lq=x_lr)

        D_x_i = c_skip * x + c_out * x_i.to(torch.float32)
        D_x_f = c_skip * x + c_out * x_f.to(torch.float32)

        if self.training or return_pair:
            return D_x_i, D_x_f

        return D_x_i + self.sampling_ig_lambda * (D_x_f - D_x_i)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class EDMUNetSR(nn.Module):
    """Single-image SR wrapper around SST-Real for deterministic one-step inference.

    SST-Real (``EDMUNet`` / ``SSTReal``) is a *single-step* EDM denoiser: an SR
    image is produced by bicubically upscaling the LR input, adding one shot of
    Gaussian noise at ``sigma_max``, running a single denoise pass, then cropping.
    There is NO iterative sampling loop (see ``sst/models/edm_real_unet_model.py``
    ``test()`` in the reference repo).

    This wrapper exposes a plain ``forward(lq) -> sr`` so the model can be driven
    by inference frameworks (spandrel / chaiNNer) that only pass an image. The
    trainable weights live under ``self.model`` (an ``SSTReal``), so its keys are
    ``model.*`` — byte-for-byte matching a saved ``EDMUNet`` checkpoint (whose
    ``self.model`` is also the ``SSTReal``). Non-weight EDM constants
    (``sigma_data``, ``use_skip``, ``sampling_ig_lambda``, sampling ``sigma_max``)
    are plain attributes and therefore absent from the state dict, exactly as in
    ``EDMUNet``.

    Notes:
        - Only 4x is supported (baked into ``pixel_unshuffle(4)`` + ``NNx4Upsampler``).
        - Noise is seeded (``seed``) so identical inputs give identical outputs.
          Because it is a diffusion model, run it WITHOUT tiling for best results;
          independent per-tile noise otherwise produces visible seams.
    """

    def __init__(
        self,
        *,
        scale: int = 4,
        sigma_data: float = 0.5,
        sigma_max: float = 1.0,  # sampling sigma (consistency_opt), NOT the model clamp bound
        use_skip: bool = True,
        sampling_ig_lambda: float = 1.0,
        min_size: int = 64,
        seed: Optional[int] = 0,
        **model_kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.use_skip = use_skip
        self.sampling_ig_lambda = sampling_ig_lambda
        self.min_size = min_size
        self.seed = seed
        self.model = SSTReal(**model_kwargs)

    def _denoise(
        self, x: torch.Tensor, sigma: torch.Tensor, lq: torch.Tensor
    ) -> torch.Tensor:
        # EDM preconditioning — mirrors EDMUNet.forward (inference branch).
        x = x.to(torch.float32)
        sigma = sigma.reshape(-1, 1, 1, 1).to(torch.float32)
        eps = 0.1 if self.use_skip else 0.001
        denom = (sigma / eps) ** 2 + self.sigma_data**2
        c_skip = self.sigma_data**2 / denom
        c_out = (sigma / eps) / denom.sqrt()
        c_noise = (sigma + 0.002).log() / 4
        x_i, x_f = self.model(x, c_noise.flatten(), lq=lq)
        D_x_i = c_skip * x + c_out * x_i.to(torch.float32)
        D_x_f = c_skip * x + c_out * x_f.to(torch.float32)
        return D_x_i + self.sampling_ig_lambda * (D_x_f - D_x_i)

    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        sf = self.scale
        ori_h, ori_w = lq.shape[2:]
        # Align LR to min_size so the largest window and the pixel-unshuffle
        # factor divide the (upscaled) feature map cleanly.
        pad_h = (self.min_size - ori_h % self.min_size) % self.min_size
        pad_w = (self.min_size - ori_w % self.min_size) % self.min_size
        if pad_h or pad_w:
            lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")

        lq_up = F.interpolate(lq.to(torch.float32), scale_factor=sf, mode="bicubic")

        if self.seed is not None:
            gen = torch.Generator(device=lq_up.device).manual_seed(self.seed)
            latent = torch.randn(
                lq_up.shape, generator=gen, device=lq_up.device, dtype=lq_up.dtype
            )
        else:
            latent = torch.randn_like(lq_up)
        x = lq_up + self.sigma_max * latent

        sigma = torch.as_tensor(
            self.sigma_max, device=lq_up.device, dtype=torch.float32
        )
        sr = self._denoise(x, sigma, lq)
        return sr[:, :, : ori_h * sf, : ori_w * sf]


# =============================================================================
# Model Factory Functions
# =============================================================================
#
# Each SR factory mirrors an ``options/`` YAML config. The "_plus" variants only
# widen ``window_sizes`` (larger receptive field), leaving all parameter shapes
# unchanged. All released SST checkpoints use attn_type='RIB'.

# Non-plus windows: mixed local/global. Plus windows: uniformly wider.
_WINDOWS_STD = [16, 32, 64, 16, 32, 64]
_WINDOWS_PLUS = [16, 32, 48, 32, 48, 96]
_WINDOWS_LIGHT = [8, 16, 32, 16, 32, 64]


def sst_light(scale: int = 4, attn_type: ATTN_TYPE = "RIB", **kwargs) -> SST:
    """SSTlight: dim=48, 5 blocks, 3 heads, exp_ratio=1.5, direct pixelshuffle."""
    return SST(
        dim=48,
        window_sizes=_WINDOWS_LIGHT,
        num_heads=[3, 3, 3, 3, 3, 3],
        ranks=[16, 16, 16, 24, 24, 24],
        n_blocks=5,
        exp_ratio=1.5,
        attn_type=attn_type,
        rib_hidden_dim=32,
        rib_n_freqs=10,
        upscaling_factor=scale,
        upsampler_type="pixelshuffle_direct",
        gate_type="DWC",
        **kwargs,
    )


def sst_light_plus(scale: int = 4, attn_type: ATTN_TYPE = "RIB", **kwargs) -> SST:
    """SSTlight_Plus: SSTlight with wider windows [16,32,48,32,48,96]."""
    return SST(
        dim=48,
        window_sizes=_WINDOWS_PLUS,
        num_heads=[3, 3, 3, 3, 3, 3],
        ranks=[16, 16, 16, 24, 24, 24],
        n_blocks=5,
        exp_ratio=1.5,
        attn_type=attn_type,
        rib_hidden_dim=32,
        rib_n_freqs=10,
        upscaling_factor=scale,
        upsampler_type="pixelshuffle_direct",
        gate_type="DWC",
        **kwargs,
    )


def sst_base(scale: int = 4, attn_type: ATTN_TYPE = "RIB", **kwargs) -> SST:
    """SST (base): dim=180, 6 blocks, 6 heads, exp_ratio=1.25, pixelshuffle."""
    return SST(
        dim=180,
        window_sizes=_WINDOWS_STD,
        num_heads=[6, 6, 6, 6, 6, 6],
        ranks=[18, 18, 18, 34, 34, 34],
        n_blocks=6,
        exp_ratio=1.25,
        attn_type=attn_type,
        rib_hidden_dim=32,
        rib_n_freqs=10,
        upscaling_factor=scale,
        upsampler_type="pixelshuffle",
        gate_type="DWC",
        **kwargs,
    )


def sst_base_plus(scale: int = 4, attn_type: ATTN_TYPE = "RIB", **kwargs) -> SST:
    """SST_Plus: SST base with wider windows [16,32,48,32,48,96]."""
    return SST(
        dim=180,
        window_sizes=_WINDOWS_PLUS,
        num_heads=[6, 6, 6, 6, 6, 6],
        ranks=[18, 18, 18, 34, 34, 34],
        n_blocks=6,
        exp_ratio=1.25,
        attn_type=attn_type,
        rib_hidden_dim=32,
        rib_n_freqs=10,
        upscaling_factor=scale,
        upsampler_type="pixelshuffle",
        gate_type="DWC",
        **kwargs,
    )


def sst_large(scale: int = 4, attn_type: ATTN_TYPE = "RIB", **kwargs) -> SST:
    """SSTLarge ("SST-L"): dim=192, 8 blocks, 6 heads, exp_ratio=2, intermediate_dim=96."""
    return SST(
        dim=192,
        window_sizes=_WINDOWS_STD,
        num_heads=[6, 6, 6, 6, 6, 6],
        ranks=[16, 16, 16, 32, 32, 32],
        n_blocks=8,
        exp_ratio=2,
        attn_type=attn_type,
        rib_hidden_dim=32,
        rib_n_freqs=10,
        upscaling_factor=scale,
        upsampler_type="pixelshuffle",
        gate_type="DWC",
        intermediate_dim=96,
        **kwargs,
    )


def sst_large_plus(scale: int = 4, attn_type: ATTN_TYPE = "RIB", **kwargs) -> SST:
    """SSTLarge_Plus: SSTLarge with wider windows [16,32,48,32,48,96]."""
    return SST(
        dim=192,
        window_sizes=_WINDOWS_PLUS,
        num_heads=[6, 6, 6, 6, 6, 6],
        ranks=[16, 16, 16, 32, 32, 32],
        n_blocks=8,
        exp_ratio=2,
        attn_type=attn_type,
        rib_hidden_dim=32,
        rib_n_freqs=10,
        upscaling_factor=scale,
        upsampler_type="pixelshuffle",
        gate_type="DWC",
        intermediate_dim=96,
        **kwargs,
    )


def sst_xl_plus(scale: int = 4, attn_type: ATTN_TYPE = "RIB", **kwargs) -> SST:
    """SSTXL_Plus ("SST-XL+"): dim=224, 10 blocks, 7 heads, exp_ratio=2, wide windows."""
    return SST(
        dim=224,
        window_sizes=_WINDOWS_PLUS,
        num_heads=[7, 7, 7, 7, 7, 7],
        ranks=[16, 16, 16, 32, 32, 32],
        n_blocks=10,
        exp_ratio=2,
        attn_type=attn_type,
        rib_hidden_dim=32,
        rib_n_freqs=10,
        upscaling_factor=scale,
        upsampler_type="pixelshuffle",
        gate_type="DWC",
        intermediate_dim=96,
        **kwargs,
    )


def sst_real(
    scale: int = 4, sampling_ig_lambda: float = 1.0, use_skip: bool = True, **kwargs
) -> EDMUNet:
    """SSTReal_X4: EDMUNet diffusion model (dim=192, 14 blocks, exp_ratio=3, drop_p=0.05).

    NOTE: forward signature is ``(x, sigma, lq)`` returning an image pair — this is
    a diffusion model, not a drop-in for the standard SR pipeline.
    """
    return EDMUNet(
        img_resolution=192,
        img_channels=3,
        scale=scale,
        sigma_data=0.5,
        sigma_min=0,
        use_skip=use_skip,
        sampling_ig_lambda=sampling_ig_lambda,
        dim=192,
        intermediate_dim=96,
        emb_channels=768,
        noise_channels=192,
        num_blocks=14,
        fuse_lr_till=6,
        ig_idx=6,
        attn_type="RIB",
        window_sizes=_WINDOWS_STD,
        num_heads=[6, 6, 6, 6, 6, 6],
        ranks=[16, 16, 16, 32, 32, 32],
        rib_hidden_dim=32,
        rib_n_freqs=10,
        exp_ratio=3,
        gate_type="DWC",
        drop_p=0.05,
        **kwargs,
    )


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    print("SST Architecture Definitions:")
    variants = [
        ("sst_light", sst_light),
        ("sst_light_plus", sst_light_plus),
        ("sst_base", sst_base),
        ("sst_base_plus", sst_base_plus),
        ("sst_large", sst_large),
        ("sst_large_plus", sst_large_plus),
        ("sst_xl_plus", sst_xl_plus),
    ]

    # RIB uses FlashAttention/bfloat16 kernels that need CUDA. On CPU we fall
    # back to 'SDPA' just for the parameter-count sanity check.
    build_attn = "RIB" if torch.cuda.is_available() else "SDPA"
    for name, fn in variants:
        model = fn(scale=4, attn_type=build_attn)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:16s}: {n_params / 1e6:7.3f} M params")

    real = sst_real(scale=4)
    n_params = sum(p.numel() for p in real.parameters() if p.requires_grad)
    print(f"  {'sst_real':16s}: {n_params / 1e6:7.3f} M params (EDMUNet diffusion)")

    if torch.cuda.is_available():
        print("\nForward smoke test (sst_light, RIB, CUDA)...")
        model = sst_light(scale=4, attn_type="RIB").cuda().eval()
        x = torch.randn(1, 3, 64, 64, device="cuda")
        with torch.inference_mode():
            y = model(x)
        print(f"  in {tuple(x.shape)} -> out {tuple(y.shape)}")
    else:
        print("\nForward smoke test (sst_light, SDPA, CPU)...")
        model = sst_light(scale=4, attn_type="SDPA").eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.inference_mode():
            y = model(x)
        print(f"  in {tuple(x.shape)} -> out {tuple(y.shape)}")
