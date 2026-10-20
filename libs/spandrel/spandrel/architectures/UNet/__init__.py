# __init__.py - UNetArch for Spandrel auto-registry (detect/load from state_dict)
from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    ModelTiling,
    SizeRequirements,
    StateDict,
)
from .__arch.UNET import UNet  # Import model (like .__arch.SCUNet)


class UNetArch(Architecture[UNet]):
    def __init__(self) -> None:
        super().__init__(
            id="UNet",
            detect=KeyCondition.has_all(
                "enc_layers.0.0.weight",   # Signature encoder key
                "bottleneck.0.weight",     # Bottleneck
                "dec_blocks.0.0.weight",   # Signature decoder key
                "final_conv.weight",       # Final output
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[UNet]:
        # Infer params from state_dict (like SCUNet's dim/config inference)
        in_nc = state_dict["enc_layers.0.0.weight"].shape[1]  # Input channels
        out_nc = state_dict["final_conv.weight"].shape[0]     # Output channels
        base_channels = state_dict["final_conv.weight"].shape[1]  # Base feat channels

        # Infer num_levels: Count unique enc_layers.N indices (like get_seq_len but for ModuleList)
        enc_keys = [k for k in state_dict if k.startswith('enc_layers.')]
        num_levels = len(set(k.split('.')[1] for k in enc_keys)) if enc_keys else 4  # Default fallback

        model = UNet(
            in_nc=in_nc,
            out_nc=out_nc,
            base_channels=base_channels,
            num_levels=num_levels,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",  # Like SCUNet; tags for binarization
            tags=["segmentation", "binarization"],
            supports_half=True,
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(minimum=32),  # Smaller than SCUNet for docs
            tiling=ModelTiling.DISCOURAGED,
        )


__all__ = ["UNetArch", "UNet"]