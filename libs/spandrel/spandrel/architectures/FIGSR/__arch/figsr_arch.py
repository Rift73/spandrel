from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# FIGSR's upsampler stack (UniUpsampleV3 with DySample, LDA_AQU and PA) is
# GFISRv2's, line for line
from spandrel.architectures.GFISRv2.__arch.gfisrv2 import SampleMods, UniUpsampleV3
from spandrel.util import store_hyperparameters


def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, (
        "The size of the first dimension: "
        f"tensor.shape[0] = {tensor.shape[0]}"
        " is not divisible by square of upscale_factor: "
        f"upscale_factor = {upscale_factor}"
    )
    sub_kernel = torch.empty(
        tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:]
    )
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.offset = nn.Parameter(torch.zeros(dim))
        self.eps = nn.Parameter(torch.Tensor(torch.ones(1) * eps), requires_grad=False)
        self.rms = nn.Parameter(
            torch.Tensor(torch.ones(1) * (dim**-0.5)), requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        norm_x = torch.addcmul(self.eps, x.norm(2, dim=1, keepdim=True), self.rms)
        return torch.addcmul(
            self.offset[:, None, None], x.div(norm_x), self.scale[:, None, None]
        )


class CustomRFFT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        return torch.view_as_real(y)

    @staticmethod
    def symbolic(g, x: torch.Value):
        shp = g.op("Shape", x)
        iH = g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64))
        iW = g.op("Constant", value_t=torch.tensor([3], dtype=torch.int64))
        nH = g.op("Gather", shp, iH, axis_i=0)
        nW = g.op("Gather", shp, iW, axis_i=0)

        axes_last = g.op("Constant", value_t=torch.tensor([4], dtype=torch.int64))
        x_u = g.op("Unsqueeze", x, axes_last)
        zero = g.op("Sub", x_u, x_u)
        x_c = g.op("Concat", x_u, zero, axis_i=4)

        Hf = g.op("Cast", nH, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        Wf = g.op("Cast", nW, to_i=torch.onnx.TensorProtoDataType.FLOAT)

        y = g.op("DFT", x_c, nW, axis_i=3, onesided_i=1)
        y = g.op("Div", y, g.op("Sqrt", Wf))

        y = g.op("DFT", y, nH, axis_i=2, onesided_i=0)
        y = g.op("Div", y, g.op("Sqrt", Hf))

        return y


class CustomIRFFT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_ri: torch.Tensor):
        x_c = torch.view_as_complex(x_ri)
        return torch.fft.irfft2(x_c, dim=(2, 3), norm="ortho")

    @staticmethod
    def symbolic(g, x: torch.Value):
        shp = g.op("Shape", x)
        iH = g.op("Constant", value_t=torch.tensor([2], dtype=torch.int64))
        iWr = g.op("Constant", value_t=torch.tensor([3], dtype=torch.int64))
        nH = g.op("Gather", shp, iH, axis_i=0)
        nWr = g.op("Gather", shp, iWr, axis_i=0)

        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.int64))
        nW = g.op("Mul", g.op("Sub", nWr, one), two)
        Hf = g.op("Cast", nH, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        Wf = g.op("Cast", nW, to_i=torch.onnx.TensorProtoDataType.FLOAT)

        yH = g.op("DFT", x, nH, axis_i=2, inverse_i=1, onesided_i=0)
        yH = g.op("Mul", yH, g.op("Sqrt", Hf))

        start = g.op("Sub", nWr, two)
        start = g.op(
            "Squeeze",
            start,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )
        limit = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        step = g.op("Constant", value_t=torch.tensor(-1, dtype=torch.int64))
        idx_r = g.op("Range", start, limit, step)

        mirW = g.op("Gather", yH, idx_r, axis_i=3)
        maskW = g.op("Constant", value_t=torch.tensor([1.0, -1.0], dtype=torch.float32))
        maskW = g.op(
            "Unsqueeze",
            maskW,
            g.op("Constant", value_t=torch.tensor([0, 1, 2, 3], dtype=torch.int64)),
        )
        mirWc = g.op("Mul", mirW, maskW)
        x_full = g.op("Concat", yH, mirWc, axis_i=3)

        y = g.op("DFT", x_full, nW, axis_i=3, inverse_i=1, onesided_i=0)
        y = g.op("Mul", y, g.op("Sqrt", Wf))

        s0 = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
        s1 = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
        axC = g.op("Constant", value_t=torch.tensor([4], dtype=torch.int64))
        y = g.op("Slice", y, s0, s1, axC)
        y = g.op("Squeeze", y, axC)

        return y


class CustomRfft2Wrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if self.training:
            y = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
            return torch.view_as_real(y)
        else:
            return CustomRFFT2.apply(x)


class CustomIrfft2Wrap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if self.training:
            x_c = torch.view_as_complex(x)  # [B,C,H,Wr]
            return torch.fft.irfft2(x_c, dim=(2, 3), norm="ortho")  # [B,C,H,W]
        else:
            return CustomIRFFT2.apply(x)


class FourierUnit(nn.Module):
    def __init__(self, in_channels: int = 48, out_channels: int = 48) -> None:
        super().__init__()
        self.rn = RMSNorm(out_channels * 2)
        self.post_norm = RMSNorm(out_channels)

        self.fdc = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            bias=True,
        )

        self.fpe = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=in_channels * 2,
            kernel_size=3,
            padding=1,
            groups=in_channels * 2,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.irfft2 = CustomIrfft2Wrap()
        self.rfft2 = CustomRfft2Wrap()

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        ffted = self.rfft2(x)
        ffted = ffted.permute(0, 4, 1, 2, 3).contiguous()
        ffted = ffted.view(b, c * 2, h, -1).to(orig_dtype)
        ffted = self.rn(ffted)
        ffted = self.fpe(ffted) + ffted
        ffted = self.fdc(ffted)
        ffted = self.gelu(ffted)
        ffted = ffted.view(b, c, 2, h, -1).permute(0, 1, 3, 4, 2).contiguous().float()
        out = self.irfft2(ffted)
        out = self.post_norm(out.to(orig_dtype))
        return out


class InceptionConv2d(nn.Module):
    """Inception convolution"""

    def __init__(
        self,
        fu_dim: int = 24,
        gc: int = 8,
        square_kernel_size: int = 13,
        band_kernel_size: int = 17,
    ) -> None:
        super().__init__()

        self.fu = FourierUnit(fu_dim, fu_dim)
        self.convhw = nn.Conv2d(
            gc, gc, square_kernel_size, padding=square_kernel_size // 2
        )
        self.convw = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
        )
        self.convh = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
        )

    def forward(
        self, x: Tensor, x_hw: Tensor, x_w: Tensor, xh: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.fu(x), self.convhw(x_hw), self.convw(x_w), self.convh(xh)


class GatedCNNBlock(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        expansion_ratio: float = 8 / 3,
        gc: int = 8,
        square_kernel_size: int = 13,
        band_kernel_size: int = 17,
    ) -> None:
        super().__init__()
        hidden = int(expansion_ratio * dim) // 8 * 8
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.act = nn.SiLU()
        self.split_indices = [hidden, hidden - dim, dim - gc * 3, gc, gc, gc]
        self.conv = InceptionConv2d(
            dim - gc * 3, gc, square_kernel_size, band_kernel_size
        )
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)

    def gated_forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        g, i, c, c_hw, c_w, c_h = torch.split(x, self.split_indices, dim=1)
        c, c_hw, c_w, c_h = self.conv(c, c_hw, c_w, c_h)
        x = self.fc2(self.act(g) * torch.cat((i, c, c_hw, c_w, c_h), dim=1))
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.gated_forward(x) + x


@store_hyperparameters()
class FIGSR(nn.Module):
    """Fourier Inception Gated Super Resolution"""

    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc: int = 3,
        dim: int = 48,
        expansion_ratio: float = 8 / 3,
        scale: int = 4,
        out_nc: int = 3,
        upsampler: SampleMods = "pixelshuffledirect",
        mid_dim: int = 32,
        n_blocks: int = 24,
        gc: int = 8,
        square_kernel_size: int = 13,
        band_kernel_size: int = 17,
    ) -> None:
        super().__init__()
        self.in_to_dim = nn.Conv2d(in_nc, dim, 3, 1, 1)
        self.pad = 2
        self.gfisr_body_half = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim, expansion_ratio, gc, square_kernel_size, band_kernel_size
                )
                for _ in range(n_blocks // 2)
            ]
        )
        self.gfisr_body_half_2 = nn.Sequential(
            *[
                GatedCNNBlock(
                    dim, expansion_ratio, gc, square_kernel_size, band_kernel_size
                )
                for _ in range(n_blocks - n_blocks // 2)
            ]
            + [nn.Conv2d(dim, dim, 3, 1, 1)]
        )
        self.cat_to_dim = nn.Conv2d(dim * 3, dim, 1)
        self.upscale = UniUpsampleV3(
            upsampler, scale, dim, out_nc, mid_dim, dysample_end_kernel=3
        )
        if upsampler == "pixelshuffledirect":
            weight = ICNR(
                self.upscale[0].weight,
                initializer=nn.init.kaiming_normal_,
                upscale_factor=scale,
            )
            self.upscale[0].weight.data.copy_(weight)

        self.scale = scale
        self.shift = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.5, requires_grad=True)
        self.scale_norm = nn.Parameter(torch.ones(1, 3, 1, 1) / 6, requires_grad=True)

    # Upstream defaults to assign=True, which hands the model the state dict's
    # own tensors (and dtype); copy into the model's tensors like every other
    # architecture here.
    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict["upscale.MetaUpsample"] = self.upscale.MetaUpsample
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.shift) / self.scale_norm

        _, _, H, W = x.shape
        mod_pad_h = (self.pad - H % self.pad) % self.pad
        mod_pad_w = (self.pad - W % self.pad) % self.pad
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        x = self.in_to_dim(x)
        x0 = self.gfisr_body_half(x)
        x1 = self.gfisr_body_half_2(x0)

        x = self.cat_to_dim(torch.cat([x1, x, x0], dim=1))
        x = self.upscale(x)[:, :, : H * self.scale, : W * self.scale]
        return x * self.scale_norm + self.shift
