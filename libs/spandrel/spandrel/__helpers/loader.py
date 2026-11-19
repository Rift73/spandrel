from __future__ import annotations

import importlib
import os
import pickle
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import load_file

from .canonicalize import canonicalize_state_dict
from .main_registry import MAIN_REGISTRY
from .model_descriptor import ModelDescriptor, StateDict
from .registry import ArchRegistry
from .unpickler import RestrictedUnpickle, ScriptObject, TorchScriptUnpickle

# `torch.load` hands any TorchScript archive to `torch.jit.load`, whatever
# `pickle_module` it is given, and has no public way to turn that off. Its own steps
# are used instead, looked up by name because strict type checking rejects private
# members.
_serialization = importlib.import_module("torch.serialization")
_SERIALIZATION_STEPS = (
    "_is_zipfile",
    "_legacy_load",
    "_open_zipfile_reader",
    "_is_torchscript_zip",
    "_load",
)

_TORCHSCRIPT_REFUSED = (
    "This TorchScript file holds more than plain weights, and TorchScript model files"
    " are not run because they can execute code. Re-save the weights as a plain"
    " PyTorch checkpoint (torch.save of the state dict) or as .safetensors."
)


class ModelLoader:
    """Class for automatically loading a pth file into any architecture"""

    def __init__(
        self,
        device: str | torch.device | None = None,
        registry: ArchRegistry = MAIN_REGISTRY,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device: torch.device = device or torch.device("cpu")
        self.registry: ArchRegistry = registry
        """
        The architecture registry to use for loading models.

        *Note:* Unless initialized with a custom registry, this is the global main registry (`MAIN_REGISTRY`).
        Modifying this registry will affect all `ModelLoader` instances without a custom registry.
        """

    def load_from_file(self, path: str | Path) -> ModelDescriptor:
        """
        Load a model from the given file path.

        Throws a `ValueError` if the file extension is not supported.
        Throws an `UnsupportedModelError` if the model architecture is not supported.
        """

        state_dict = self.load_state_dict_from_file(path)
        return self.load_from_state_dict(state_dict)

    def load_state_dict_from_file(self, path: str | Path) -> StateDict:
        """
        Load the state dict of a model from the given file path.

        State dicts are typically only useful to pass them into the `load`
        function of a specific architecture.

        Throws a `ValueError` if the file extension is not supported.
        """

        extension = os.path.splitext(path)[1].lower()

        state_dict: StateDict
        if extension == ".pt" or extension == ".pth" or extension == ".ckpt":
            state_dict = self._load_pth(path)
        elif extension == ".safetensors":
            state_dict = self._load_safetensors(path)
        else:
            raise ValueError(
                f"Unsupported model file extension {extension}. Please try a supported model type."
            )

        return canonicalize_state_dict(state_dict)

    def load_from_state_dict(self, state_dict: StateDict) -> ModelDescriptor:
        """
        Load a model from the given state dict.

        Throws an `UnsupportedModelError` if the model architecture is not supported.
        """

        return self.registry.load(state_dict).to(self.device)

    def _load_pth(self, path: str | Path) -> StateDict:
        # The same steps as `torch.load`, except that a TorchScript archive is read
        # by `_load_torchscript` and never by `torch.jit.load`.
        for step in _SERIALIZATION_STEPS:
            if not hasattr(_serialization, step):
                raise ValueError(
                    f"This version of PyTorch ({torch.__version__}) is not supported for"
                    f" loading .pt, .pth and .ckpt files: torch.serialization.{step} is"
                    " missing."
                )
        with open(path, "rb") as file:
            if not _serialization._is_zipfile(file):
                return _serialization._legacy_load(
                    file, self.device, RestrictedUnpickle, encoding="utf-8"
                )
            with _serialization._open_zipfile_reader(file) as zip_file:
                if _serialization._is_torchscript_zip(zip_file):
                    return self._load_torchscript(zip_file)
                return _serialization._load(
                    zip_file, self.device, RestrictedUnpickle, encoding="utf-8"
                )

    def _load_torchscript(self, zip_file: object) -> StateDict:
        # Only the archive's pickled modules are read, as `ScriptObject`s; its
        # code is not compiled or run. The state dict is every tensor that the
        # modules hold, named by its attribute path.
        try:
            root = _serialization._load(
                zip_file, self.device, TorchScriptUnpickle, encoding="utf-8"
            )
        except pickle.UnpicklingError as error:
            raise ValueError(_TORCHSCRIPT_REFUSED) from error
        return _collect_script_tensors(root)

    def _load_safetensors(self, path: str | Path) -> StateDict:
        return load_file(path, device=str(self.device))


# Bounds of the walk below, which reads a file-controlled object graph. Modules that
# the pickle shares (one module under two names, tied weights) are visited once per
# name, as by `state_dict()`, so without bounds a small crafted archive could ask
# for 2**n visits, an endless walk or keys of any length. Real modules nest less
# than 20 deep and hold a few thousand modules and tensors.
_MAX_SCRIPT_DEPTH = 64
_MAX_SCRIPT_VISITS = 100_000
_MAX_SCRIPT_KEY_CHARS = 16_000_000


def _script_state(obj: object) -> dict[object, object]:
    # A module that pickles its own state (a TorchScript __getstate__) can only be
    # restored by running its __setstate__, so it is refused.
    if not isinstance(obj, ScriptObject) or not isinstance(obj.state, dict):
        raise ValueError(_TORCHSCRIPT_REFUSED)
    return obj.state


def _collect_script_tensors(root: object) -> StateDict:
    # Depth first in attribute order, which is the order of `state_dict()`. The
    # path is kept as its parts and joined only for a tensor. An OrderedDict, as
    # `state_dict()` gives.
    state_dict: StateDict = OrderedDict()
    path: list[str] = []
    stack: list[Iterator[tuple[object, object]]] = [iter(_script_state(root).items())]
    visits = 0
    key_chars = 0
    while stack:
        item = next(stack[-1], None)
        if item is None:
            stack.pop()
            if path:
                path.pop()
            continue
        name, value = item
        if not isinstance(value, (torch.Tensor, ScriptObject)):
            continue
        visits += 1
        if visits > _MAX_SCRIPT_VISITS:
            raise ValueError(_TORCHSCRIPT_REFUSED)
        if isinstance(value, torch.Tensor):
            key = ".".join([*path, str(name)])
            key_chars += len(key)
            if key_chars > _MAX_SCRIPT_KEY_CHARS:
                raise ValueError(_TORCHSCRIPT_REFUSED)
            state_dict[key] = value
            continue
        # also ends the walk through an object that holds itself
        if len(stack) >= _MAX_SCRIPT_DEPTH:
            raise ValueError(_TORCHSCRIPT_REFUSED)
        path.append(str(name))
        stack.append(iter(_script_state(value).items()))
    return state_dict
