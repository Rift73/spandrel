from __future__ import annotations

from typing import Mapping
import torch

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
)
from .__arch.LAWFFT import LAWFFT, SampleMods


class LAWFFTArch(Architecture[LAWFFT]):
    def __init__(self):
        super().__init__(
            id="LAWFFT",
            detect=KeyCondition.has_all(
                KeyCondition.has_any(
                    "in_to_dim.weight",
                    "in_to_dim.1.weight",
                ),
                "body.0.residual.0.token_mix.1.local.0.kernel_gen.1.bias",
                "body.0.residual.0.token_mix.1.att.to_hidden.bias",
                "body.0.residual.0.channel_mix1.1.project_in.bias",
            ),
        )

    def load(self, state_dict: Mapping[str, object]) -> ImageModelDescriptor[LAWFFT]:
        # --- Извлечение параметров из state_dict ---
        samplemods = ["conv", "pixelshuffledirect", "pixelshuffle", "nearest+conv", "dysample"]
        
        # Обработка MetaUpsample
        meta_upsample = state_dict["upscale.MetaUpsample"]
        if isinstance(meta_upsample, torch.Tensor):
            meta_values = [i.item() for i in meta_upsample]
        else:
            meta_values = list(meta_upsample)
        
        _version, upsampler_idx, scale, dim, in_ch, mid_dim, _group = meta_values
        upsampler = samplemods[int(upsampler_idx)]
        
        unshuffle_mod = "in_to_dim.1.weight" in state_dict
        window_size = state_dict["window_size"].item()
        
        # Расчет параметров архитектуры
        local_dim = state_dict["body.0.residual.0.token_mix.1.local.0.kernel_gen.1.bias"].shape[0]
        split = 1 / (dim / local_dim)
        n_rblock = get_seq_len(state_dict, "body")
        n_mblock = get_seq_len(state_dict, "body.0.residual") - 1
        global_dim = dim - int((dim * split))
        t_mid_factor = state_dict["body.0.residual.1.token_mix.1.att.to_hidden.bias"].shape[0] / global_dim / 3
        mlp_factor = state_dict["body.0.residual.1.channel_mix1.1.project_in.bias"].shape[0] / dim / 2

        # --- Создание модели ---
        model = LAWFFT(
            in_ch=in_ch,
            dim=dim,
            split=split,
            scale=scale,
            n_rblock=n_rblock,
            n_mblock=n_mblock,
            t_mid_factor=t_mid_factor,
            window_size=window_size,
            mlp_factor=mlp_factor,
            unshuffle_mod=unshuffle_mod,
            upsampler=upsampler,
            mid_dim=mid_dim,
        )
        
        # --- Создание дескриптора ---
        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}dim", f"{n_rblock}rb", f"{n_mblock}mb"],
            supports_half=False,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_ch,
            output_channels=in_ch,
            size_requirements=SizeRequirements(multiple_of=model.pad),
        )


__all__ = ["LAWFFTArch", "LAWFFT"]
