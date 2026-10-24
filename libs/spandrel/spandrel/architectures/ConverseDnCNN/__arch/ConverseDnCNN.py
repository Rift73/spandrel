#!/usr/bin/env python3
# type: ignore
"""ConverseDnCNN architecture implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spandrel.util import store_hyperparameters


class Converse2D(nn.Module):
    """Converse2D module for reverse convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.rand(1, channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for Converse2D."""
        _, _, h, w = y.shape

        if h < self.kernel_size or w < self.kernel_size:
            return y

        kernel = self.weight
        lambda_val = F.sigmoid(self.bias - 9.0) + self.eps

        fft_h = 1 << (h - 1).bit_length()
        fft_w = 1 << (w - 1).bit_length()

        padded_y = F.pad(y, (0, fft_w - w, 0, fft_h - h), mode="replicate")
        padded_kernel = F.pad(
            kernel, (0, fft_w - self.kernel_size, 0, fft_h - self.kernel_size)
        )

        f_y = torch.fft.fft2(padded_y, norm="ortho")
        f_k = torch.fft.fft2(padded_kernel, norm="ortho")

        f_x0 = f_y
        conj_f_k = f_k.conj()

        denom = torch.abs(f_k) ** 2 + lambda_val
        numerator = conj_f_k * f_y + lambda_val * f_x0
        f_x = numerator / denom

        x = torch.fft.ifft2(f_x, norm="ortho").real
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        return x[:, :, :h, :w]


class ReverseConvBlock(nn.Module):
    """Reverse convolution block."""

    def __init__(
        self,
        nc: int,
        expansion_ratio: int = 2,
        kernel_size: int = 7,
    ):
        super().__init__()
        hidden = nc * expansion_ratio
        self.act = nn.GELU()

        self.conv1 = nn.Module()
        self.conv1.add_module("0", nn.LayerNorm(nc))
        self.conv1.add_module("1", nn.Conv2d(nc, hidden, 1, bias=True))
        self.conv1.add_module("3", Converse2D(hidden, kernel_size=kernel_size))
        self.conv1.add_module("5", nn.Conv2d(hidden, nc, 1, bias=True))

        self.conv2 = nn.Module()
        self.conv2.add_module("0", nn.LayerNorm(nc))
        self.conv2.add_module("1", nn.Conv2d(nc, hidden, 1, bias=True))
        self.conv2.add_module("3", nn.Conv2d(hidden, nc, 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ReverseConvBlock."""
        identity = x

        x1 = x.permute(0, 2, 3, 1)
        x1 = self.conv1._modules['0'](x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.conv1._modules['1'](x1)
        x1 = self.act(x1)
        x1 = self.conv1._modules['3'](x1)
        x1 = self.act(x1)
        x1 = self.conv1._modules['5'](x1)

        x2 = x.permute(0, 2, 3, 1)
        x2 = self.conv2._modules['0'](x2)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = self.conv2._modules['1'](x2)
        x2 = self.act(x2)
        x2 = self.conv2._modules['3'](x2)

        return x1 + x2 + identity


@store_hyperparameters()
class ConverseDnCNN(nn.Module):
    """ConverseDnCNN model."""

    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc: int = 1,
        out_nc: int = 1,
        nc: int = 64,
        nb: int = 20,
        expansion_ratio: int = 2,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, nc, 1, bias=True))
        self.act = nn.GELU()
        self.m_body = nn.Sequential(
            *[
                ReverseConvBlock(
                    nc, expansion_ratio=expansion_ratio, kernel_size=kernel_size
                )
                for _ in range(nb)
            ]
        )
        self.m_tail = nn.Sequential(nn.Conv2d(nc, out_nc, 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ConverseDnCNN."""
        # Адаптер диапазона: [0, 1] -> [0, 255]
        x_scaled = x * 255.0

        h = self.act(self.m_head(x_scaled))
        h = self.m_body(h)
        h = self.m_tail(h)
        out_scaled = x_scaled - h

        # Обратное преобразование: [0, 255] -> [0, 1]
        out = out_scaled / 255.0

        return torch.clamp(out, 0, 1)