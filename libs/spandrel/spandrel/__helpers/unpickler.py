# Safe unpickler to prevent arbitrary code execution
import collections
import pickle
from types import FunctionType, SimpleNamespace
from typing import Callable

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
        found = super().find_class(module, name)
        # A pickle can set attributes on the globals it is given (BUILD), which
        # would change them for the rest of the process. Functions are therefore
        # given as a new function each time, and typing's OrderedDict alias as the
        # class it stands for.
        if module == "typing":
            return collections.OrderedDict
        if isinstance(found, FunctionType):
            function: Callable[..., object] = found
            return lambda *args: function(*args)
        return found


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


class TorchScriptUnpickler(RestrictedUnpickler):
    """
    Reads the pickled modules of a TorchScript archive as `ScriptObject`s, so
    that only tensors and plain values are built from the file.
    """

    @override
    def find_class(self, module: str, name: str):
        # New each time, for the same reason as in RestrictedUnpickler
        if module == "__torch__" or module.startswith("__torch__."):
            return type(name, (ScriptObject,), {})
        if module == "torch.jit._pickle" and name in _script_list_functions:
            return lambda value, *_: value
        return super().find_class(module, name)


TorchScriptUnpickle = SimpleNamespace(
    Unpickler=TorchScriptUnpickler,
    __name__="pickle",
    load=lambda *args, **kwargs: TorchScriptUnpickler(*args, **kwargs).load(),
)
