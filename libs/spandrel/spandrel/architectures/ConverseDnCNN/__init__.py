from __future__ import annotations

from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)

from .__arch.ConverseDnCNN import ConverseDnCNN


class ConverseDnCNNArch(Architecture[ConverseDnCNN]):
    """Architecture class for ConverseDnCNN."""

    def __init__(self) -> None:
        super().__init__(
            id="ConverseDnCNN",
            detect=KeyCondition.has_all(
                "m_head.0.weight",
                "m_body.0.conv1.0.weight",
                "m_body.0.conv1.1.weight",
                "m_body.0.conv1.3.weight",
                "m_body.0.conv1.5.weight",
                "m_body.0.conv2.0.weight",
                "m_body.0.conv2.1.weight",
                "m_body.0.conv2.3.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[ConverseDnCNN]:
        """Load the model from state_dict."""
        in_nc = state_dict["m_head.0.weight"].shape[1]
        nc = state_dict["m_head.0.weight"].shape[0]
        out_nc = state_dict["m_tail.0.weight"].shape[0]
        expansion_ratio = state_dict["m_body.0.conv1.1.weight"].shape[0] // nc
        kernel_size = state_dict["m_body.0.conv1.3.weight"].shape[3]

        body_keys = [k for k in state_dict if k.startswith("m_body.")]
        max_b = max(
            int(k.split(".")[1]) for k in body_keys if k.split(".")[1].isdigit()
        )
        nb = max_b + 1

        model = ConverseDnCNN(
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            nb=nb,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
        )

        tags = [f"{nc}nf", f"{nb}nb"]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=tags,
            supports_half=False,
            supports_bfloat16=True,
            scale=1,
            input_channels=in_nc,
            output_channels=out_nc,
            # Убираем size_requirements для предотвращения медленной "плиточной" обработки
        )


__all__ = ["ConverseDnCNNArch", "ConverseDnCNN"]