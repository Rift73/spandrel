# Safe unpickler to prevent arbitrary code execution
import pickle
from types import SimpleNamespace

from typing_extensions import override

safe_list = {
    ("collections", "OrderedDict"),
    ("typing", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "BFloat16Storage"),
    ("torch", "FloatStorage"),
    ("torch", "HalfStorage"),
    ("torch", "IntStorage"),
    ("torch", "LongStorage"),
    ("torch", "DoubleStorage"),
}


class RestrictedUnpickler(pickle.Unpickler):
    @override
    def find_class(self, module: str, name: str):
        # Only allow required classes to load state dict
        if (module, name) not in safe_list:
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")
        return super().find_class(module, name)


RestrictedUnpickle = SimpleNamespace(
    Unpickler=RestrictedUnpickler,
    __name__="pickle",
    load=lambda *args, **kwargs: RestrictedUnpickler(*args, **kwargs).load(),
)


class ScriptObject:
    """
    Stands in for an object of a class compiled by TorchScript. It keeps the
    pickled state and runs none of the class's code.
    """

    state: object = None

    def __setstate__(self, state: object) -> None:
        self.state = state


# TorchScript pickles typed lists through these functions, which all return
# their first argument.
_script_list_functions = {
    "build_intlist",
    "build_tensorlist",
    "build_doublelist",
    "build_boollist",
    "restore_type_tag",
}


def _first_argument(value: object, *_: object) -> object:
    return value


class TorchScriptUnpickler(RestrictedUnpickler):
    """
    Reads the pickled modules of a TorchScript archive as `ScriptObject`s, so
    that only tensors and plain values are built from the file.
    """

    @override
    def find_class(self, module: str, name: str):
        if module == "__torch__" or module.startswith("__torch__."):
            return ScriptObject
        if module == "torch.jit._pickle" and name in _script_list_functions:
            return _first_argument
        return super().find_class(module, name)


TorchScriptUnpickle = SimpleNamespace(
    Unpickler=TorchScriptUnpickler,
    __name__="pickle",
    load=lambda *args, **kwargs: TorchScriptUnpickler(*args, **kwargs).load(),
)
