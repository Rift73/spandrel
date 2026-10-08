import torch
import torch.nn.functional as F


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
    *,
    mode: str,
    value: float = 0.0,
) -> torch.Tensor:
    """
    Pad a tensor's size to a multiple of a number.

    Args:
        tensor: Tensor to pad.
        multiple: Size multiple to pad to.
        mode: Padding mode; see `torch.nn.functional.pad`.
        value: Padding value; see `torch.nn.functional.pad`.

    Returns:
        Padded tensor, or the original tensor if no padding was needed.
        During ONNX export, the tensor is always padded.
    """
    _, _, h, w = tensor.size()
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # During ONNX export, h and w are traced, and a Python branch on them would
    # freeze the example input's padding (usually none) into the graph. Always
    # pad there, so the exported graph pads every input size.
    if (multiple > 1 and torch.onnx.is_in_onnx_export()) or pad_h or pad_w:
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode, value)
    return tensor
