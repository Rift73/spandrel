from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, TypeVar

import torch
from typing_extensions import override

from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ...__helpers.rocm.common import from_triton

if TYPE_CHECKING:
    from .__arch.rdual_arch import RDUAL

logger = logging.getLogger("spandrel.rdual")

T = TypeVar("T")


# The architecture module's public names for the TensorRT export.
_EXPORT_NAMES = frozenset(
    (
        "HAS_DIRECT_ATTENTION",
        "DIRECT_ATTENTION_ERROR",
        "TEMPORAL_FORWARD_KERNEL",
        "KV_WARP_FORWARD_KERNEL",
        "SpatialAttention",
        "GROUP_INDICES",
        "NeuFlowV2",
        "NF_CONTEXT_DIM_S16",
        "NF_CONTEXT_DIM_S8",
        "NF_FEATURE_DIM_S16",
        "NF_FEATURE_DIM_S8",
        "NF_ITER_CONTEXT_DIM_S16",
        "NF_ITER_CONTEXT_DIM_S8",
        "NF_ITERS_S16",
        "NF_ITERS_S8",
    )
)


def __getattr__(name: str) -> object:
    # The architecture's module imports Triton and Flex attention and registers its
    # kernels: it loads with the first RDUAL model, not with every import of spandrel.
    if name == "RDUAL" or name in _EXPORT_NAMES:
        from .__arch import rdual_arch

        return getattr(rdual_arch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _forward(model: RDUAL, frames: torch.Tensor) -> torch.Tensor:
    cuda = frames.device.type == "cuda"
    with torch.autocast("cuda", torch.bfloat16, enabled=cuda):
        return model(frames.float()).float()


def _window(
    model: RDUAL,
    frames: torch.Tensor,
    keep: int,
    past: tuple[torch.Tensor, torch.Tensor] | None,
    flow_volume: bool,
    hold_encodes: bool,
    tile: tuple[int, int] | None,
    overlap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cuda = frames.device.type == "cuda"
    with torch.autocast("cuda", torch.bfloat16, enabled=cuda):
        outputs, state = model.infer_window(
            frames.float(),
            keep,
            past,
            flow_volume=flow_volume,
            hold_encodes=hold_encodes,
            tile=tile,
            overlap=overlap,
        )
    return outputs.float(), state


def _falling_back(model: RDUAL, run: Callable[[], T]) -> T:
    """run(), and when the Triton kernels of the temporal stage or of the
    neighborhood attention fail to compile (a Triton too old for this Python, for
    example), run() again on the PyTorch paths, which are then kept for good."""
    from .__arch import rdual_arch

    try:
        return run()
    except Exception as error:
        triton = model.temporal.backend == "auto" or rdual_arch.NA_TRITON
        if not triton or not from_triton(error):
            raise
        logger.warning(
            "RDUAL's Triton kernels failed (%s: %s); using its PyTorch path",
            type(error).__name__,
            error,
        )
        model.temporal.backend = "reference"
        rdual_arch.NA_TRITON = False
        return run()


def run_clip(model: RDUAL, frames: torch.Tensor) -> torch.Tensor:
    """B,T,3,H,W frames in 0..1 to B,T,3,sH,sW in FP32.

    On CUDA the trunk runs under BF16 autocast; the bundled flow switches autocast
    off for itself. A failing Triton temporal kernel switches the model to its
    PyTorch temporal path, and the clip runs again."""
    return _falling_back(model, lambda: _forward(model, frames))


def run_window(
    model: RDUAL,
    frames: torch.Tensor,
    keep: int,
    past: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    flow_volume: bool = True,
    hold_encodes: bool = True,
    tile: tuple[int, int] | None = None,
    overlap: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RDUAL.infer_window() with run_clip's precision and Triton fallback: the
    B,keep,3,sH,sW outputs in FP32 of a window of B,n,3,H,W frames and the state
    to carry, from the (state, previous frame) `past` of the window before it or a
    cold start. flow_volume, hold_encodes, tile and overlap are infer_window()'s
    memory modes."""
    return _falling_back(
        model,
        lambda: _window(
            model, frames, keep, past, flow_volume, hold_encodes, tile, overlap
        ),
    )


def _call(model: RDUAL, image: torch.Tensor) -> torch.Tensor:
    """One image is a clip of one frame: no neighbors, so no temporal reading."""
    return run_clip(model, image[:, None])[:, 0]


class RDUALArch(Architecture["RDUAL"]):
    def __init__(self) -> None:
        super().__init__(
            id="RDUAL",
            detect=KeyCondition.has_all(
                "flow_loaded",
                "temporal.query.weight",
                "temporal.output.weight",
                "groups.0.conv.experts",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[RDUAL]:
        from .__arch.rdual_arch import RDUAL

        # The scale is only in the shapes of the reconstruction tail.
        experts = int(state_dict["groups.0.conv.experts"].shape[0])
        for scale in (2, 4, 1):
            # every RDUAL runs its local layers and its retrieval as neighborhoods
            model = RDUAL(
                scale=scale,
                use_checkpoint=False,
                dynamic_experts=experts,
                local_na=True,
                retrieval_na=True,
            )
            reference = model.state_dict()
            if reference.keys() == state_dict.keys() and all(
                reference[key].shape == state_dict[key].shape for key in reference
            ):
                break
        else:
            raise ValueError("RDUAL: the weights match no scale of 1, 2 or 4")
        if not bool(state_dict["flow_loaded"]):
            raise ValueError("RDUAL: these weights hold no optical flow weights")

        descriptor = ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=["Light", f"{experts} experts", "PU2", "temporal"],
            supports_half=False,
            supports_bfloat16=False,
            scale=scale,
            input_channels=3,
            output_channels=3,
            size_requirements=SizeRequirements(minimum=16),
            call_fn=_call,
        )
        # Fold the reparameterized convolutions once for inference, after the strict
        # load above, in FP32 on the CPU before the loader moves the model.
        model.float().eval()
        model.prepare_for_export()
        return descriptor


__all__ = ["RDUALArch", "RDUAL", "run_clip", "run_window"]
