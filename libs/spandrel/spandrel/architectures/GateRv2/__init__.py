from typing import Mapping
import torch
from spandrel.util import KeyCondition, get_seq_len
from .__arch.GateRv2 import GateRv2, SampleMods
from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
)

"""GateRv2 architecture support for spandrel."""

class GateRv2Arch(Architecture[GateRv2]):
    """GateRv2 architecture class."""

    def __init__(self):
        super().__init__(
            id='GateRv2',
            detect=KeyCondition.has_all(
                'in_to_dim.weight',
                'encode.0.gated.0.gamma0',
                'latent.0.norm.scale',
                'decode.0.gated.0.gamma0',
                'decode.0.shor.weight',
            ),
        )

    def load(self, state_dict: Mapping[str, object]) -> ImageModelDescriptor[GateRv2]:
        """Load the model from state_dict."""
        # --- Extract parameters from state_dict ---
        in_ch = state_dict['in_to_dim.weight'].shape[1]
        dim = state_dict['in_to_dim.weight'].shape[0]

        # Determine number of encoder/decoder blocks
        enc_blocks_len = get_seq_len(state_dict, 'encode')
        dec_blocks_len = get_seq_len(state_dict, 'decode')

        enc_blocks = [get_seq_len(state_dict, f'encode.{i}.gated')
                      for i in range(enc_blocks_len)]
        dec_blocks = [get_seq_len(state_dict, f'decode.{i}.gated')
                      for i in range(dec_blocks_len)]

        num_latent = get_seq_len(state_dict, 'latent')

        scale = 1
        upsample_mid_dim = 32
        upsampler: SampleMods = 'conv'
        out_ch = in_ch

        # Handle metadata for upscale if present
        if 'upsample.MetaUpsample' in state_dict:
            meta = state_dict['upsample.MetaUpsample']
            # Convert tensor to list if not already
            if isinstance(meta, torch.Tensor):
                meta_values = [v.item() for v in meta]
            else:
                meta_values = list(meta)

            upsample_options = list(SampleMods.__args__)

            _version, upsampler_idx, scale, _in_dim, out_ch, upsample_mid_dim, _group = meta_values
            upsampler = upsample_options[int(upsampler_idx)]

        # --- Create model ---
        model = GateRv2(
            in_ch=in_ch,
            dim=dim,
            enc_blocks=tuple(enc_blocks),
            dec_blocks=tuple(dec_blocks),
            num_latent=num_latent,
            scale=scale,
            upsampler=upsampler,
            upsample_mid_dim=upsample_mid_dim,
        )

        # --- Create model descriptor ---
        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose='Restoration' if scale == 1 else 'SR',
            tags=[f'{dim}dim'],
            supports_half=False,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_ch,
            output_channels=out_ch,
            size_requirements=SizeRequirements(multiple_of=model.pad),
        )


__all__ = ["GateRv2Arch", "GateRv2"]
