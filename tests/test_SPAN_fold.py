import copy

import torch

from spandrel.architectures.SPAN import SPAN, SPANArch
from spandrel.architectures.SPAN.__arch.span import Conv3XC


def small_span():
    net = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=8, upscale=2)
    return SPANArch().load(net.state_dict()).eval()


def fold_counts(model) -> list[int]:
    """A list that grows by one with each fold in `model`."""
    folds: list[int] = []
    for module in model.model.modules():
        if isinstance(module, Conv3XC):

            def counted(update=module.update_params) -> None:
                folds.append(1)
                update()

            module.update_params = counted
    return folds


def fold_and_run(model, image: torch.Tensor) -> torch.Tensor:
    for module in model.model.modules():
        if isinstance(module, Conv3XC):
            module.update_params()
    return model(image)


def test_inference_folds_once_per_dtype():
    model = small_span()
    folds = fold_counts(model)
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        model(image)
        once = len(folds)
        assert once == 20
        model(image)
        assert len(folds) == once
        model.half()
        model(image.half())
        assert len(folds) == 2 * once


def test_the_output_is_that_of_a_fold_on_every_call():
    model = small_span()
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        output = model(image)
        assert torch.equal(output, fold_and_run(model, image))
        assert torch.equal(output, model(image))


def test_loaded_weights_fold_again():
    model = small_span()
    image = torch.rand(1, 3, 16, 16)
    other = {k: v + 0.01 for k, v in model.model.state_dict().items()}
    with torch.no_grad():
        before = model(image)
        model.model.load_state_dict(other)
        after = model(image)
        assert not torch.equal(before, after)
        assert torch.equal(after, fold_and_run(model, image))


def test_training_folds_again():
    model = small_span()
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        before = model(image)
        model.train()
        model.model(image)  # the descriptor's call would leave training mode
        for parameter in model.model.parameters():
            parameter.add_(0.01)
        model.eval()
        after = model(image)
        assert not torch.equal(before, after)
        assert torch.equal(after, fold_and_run(model, image))


def test_a_fold_in_inference_mode_keeps_normal_tensors():
    model = small_span()
    image = torch.rand(1, 3, 16, 16)
    with torch.inference_mode():
        expected = model(image)
    assert not model.model.conv_1.eval_conv.weight.is_inference()
    assert torch.equal(expected, model(image).detach())


def test_a_model_converted_in_inference_mode_folds_there():
    # An upscale copies the model to its device and dtype inside inference mode, so the
    # copy's weights are inference tensors, which a fold outside that mode cannot read.
    model = small_span()
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        expected = copy.deepcopy(model).half()(image.half())
    with torch.inference_mode():
        converted = copy.deepcopy(model).half()
        output = converted(image.half())
    assert converted.model.conv_1.eval_conv.weight.is_inference()
    assert torch.equal(expected, output)
    with torch.no_grad():
        assert torch.equal(expected, converted(image.half()))


def test_a_cast_there_and_back_folds_again():
    # The weights rounded in FP16 fold to other values than their FP32 fold rounded.
    model = small_span()
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        model(image)
        model.half()
        model.float()
        output = model(image)
        assert torch.equal(output, fold_and_run(model, image))


def test_a_fold_keeps_the_weights_memory_format():
    # A model moved to channels-last order stays in it.
    model = small_span()
    model.model.to(memory_format=torch.channels_last)
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        model(image)
    for module in model.model.modules():
        if isinstance(module, Conv3XC):
            weight = module.eval_conv.weight
            assert weight.is_contiguous(memory_format=torch.channels_last)
