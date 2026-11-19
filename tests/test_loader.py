from __future__ import annotations

import collections
import pickle
import struct
import typing
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor

from spandrel import ModelLoader
from spandrel.architectures.ESRGAN import ESRGAN


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            torch.nn.Conv2d(4, 4, 3, padding=1) for _ in range(2)
        )
        self.norm = torch.nn.BatchNorm2d(4)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x)
        return self.norm(x)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.body = torch.nn.Sequential(Block(), torch.nn.LeakyReLU(0.2), Block())
        self.tail = torch.nn.Conv2d(4, 12, 3, padding=1)
        self.up = torch.nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.tail(self.body(self.head(x))))


class WritesOnLoad(torch.nn.Module):
    """A TorchScript module whose stored __setstate__ writes into `target`."""

    def __init__(self, target: str):
        super().__init__()
        self.target = target
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        return x

    @torch.jit.export
    def __getstate__(self):
        return (self.target, self.weight, self.training)

    @torch.jit.export
    def __setstate__(self, state: tuple[str, Tensor, bool]):
        self.target = state[0]
        self.weight = state[1]
        self.training = state[2]
        written = torch.from_file(self.target, True, 8, dtype=torch.uint8)
        written.fill_(33)
        # printed so that TorchScript keeps the write
        print(written)


def make_net() -> Net:
    net = Net().eval()
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.rand_like(p))
    return net


def assert_same_state_dict(actual: dict[str, Tensor], expected: dict[str, Tensor]):
    assert list(actual) == list(expected)
    for key, value in expected.items():
        assert actual[key].dtype == value.dtype, key
        assert torch.equal(actual[key], value), key


@pytest.mark.parametrize("extension", [".pt", ".pth", ".ckpt"])
def test_torchscript_code_does_not_run(tmp_path: Path, extension: str):
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"UNTOUCHED")
    path = tmp_path / f"model{extension}"
    torch.jit.script(WritesOnLoad(str(victim))).save(str(path))
    victim.write_bytes(b"UNTOUCHED")

    with pytest.raises(ValueError, match="TorchScript model files are not run"):
        ModelLoader().load_state_dict_from_file(path)

    assert victim.read_bytes() == b"UNTOUCHED"


@pytest.mark.parametrize("kind", ["trace", "script"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_torchscript_weights_are_read(tmp_path: Path, kind: str, dtype: torch.dtype):
    net = make_net().to(dtype)
    if kind == "trace":
        module = torch.jit.trace(net, torch.rand(1, 3, 8, 8, dtype=dtype))
    else:
        module = torch.jit.script(net)
    path = tmp_path / "model.pt"
    module.save(str(path))
    expected = torch.jit.load(str(path), map_location="cpu").state_dict()

    actual = ModelLoader().load_state_dict_from_file(path)

    assert_same_state_dict(actual, expected)


def test_traced_esrgan_loads(tmp_path: Path):
    net = ESRGAN(in_nc=3, out_nc=3, num_filters=16, num_blocks=2, scale=2).eval()
    path = tmp_path / "model.pt"
    torch.jit.trace(net, torch.rand(1, 3, 8, 8)).save(str(path))

    model = ModelLoader().load_from_file(path)

    assert model.architecture.id == "ESRGAN"
    assert model.scale == 2
    assert_same_state_dict(model.model.state_dict(), net.state_dict())


class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.again = self.first  # one module under two names
        self.last = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.last.weight = self.first.weight  # tied weights

    def forward(self, x: Tensor) -> Tensor:
        return self.last(self.again(self.first(x)))


@pytest.mark.parametrize("kind", ["trace", "script"])
def test_torchscript_shared_modules_and_weights(tmp_path: Path, kind: str):
    net = Shared().eval()
    if kind == "trace":
        module = torch.jit.trace(net, torch.rand(1, 3, 8, 8))
    else:
        module = torch.jit.script(net)
    path = tmp_path / "model.pt"
    module.save(str(path))
    expected = torch.jit.load(str(path), map_location="cpu").state_dict()

    actual = ModelLoader().load_state_dict_from_file(path)

    assert_same_state_dict(actual, expected)


def _torchscript_archive(
    path: Path, data: bytes, records: tuple[tuple[str, bytes], ...] = ()
):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", data)
        for name, record in records:
            archive.writestr(f"archive/{name}", record)
        archive.writestr("archive/constants.pkl", b"\x80\x02).")
        archive.writestr("archive/version", b"3\n")


def _script_object(fields: bytes) -> bytes:
    # an object of the TorchScript class __torch__.M with the given state items
    return b"c__torch__\nM\n)\x81}" + fields + b"b"


def test_torchscript_shared_objects_are_walked_within_bounds(tmp_path: Path):
    # Each object holds the one before it twice: 2**22 paths to the first one, a
    # few seconds for a walk without a bound, which then returns an empty dict.
    data = b"\x80\x04" + _script_object(_pickled_text("v") + b"K\x01s") + b"\x94"
    for i in range(22):
        held = b"h" + bytes([i])
        fields = _pickled_text("a") + held + b"s" + _pickled_text("b") + held + b"s"
        data += _script_object(fields) + b"\x94"
    path = tmp_path / "model.pt"
    _torchscript_archive(path, data + b".")

    with pytest.raises(ValueError, match="TorchScript model files are not run"):
        ModelLoader().load_state_dict_from_file(path)


def test_torchscript_object_holding_itself(tmp_path: Path):
    data = b"\x80\x04c__torch__\nM\n)\x81\x94}" + _pickled_text("self") + b"h\x00sb."
    path = tmp_path / "model.pt"
    _torchscript_archive(path, data)

    with pytest.raises(ValueError, match="TorchScript model files are not run"):
        ModelLoader().load_state_dict_from_file(path)


def test_torchscript_nesting_is_bounded(tmp_path: Path):
    # 200 objects, each inside the one after it
    data = b"\x80\x04" + _script_object(_pickled_text("v") + b"K\x01s") + b"\x94"
    for i in range(200):
        held = b"h" + bytes([i])
        data += _script_object(_pickled_text("a") + held + b"s") + b"\x94"
    path = tmp_path / "model.pt"
    _torchscript_archive(path, data + b".")

    with pytest.raises(ValueError, match="TorchScript model files are not run"):
        ModelLoader().load_state_dict_from_file(path)


def test_torchscript_key_length_is_bounded(tmp_path: Path):
    # Seven nested objects, each holding one 4-byte tensor as "t" and the object
    # before it under one pickled 1 MB name: 21 million key characters from a
    # 1 MB file.
    tensor = b"ctorch._utils\n_rebuild_tensor_v2\n((" + _pickled_text("storage")
    tensor += b"ctorch\nFloatStorage\n" + _pickled_text("0") + _pickled_text("cpu")
    tensor += b"K\x01tQK\x00K\x01\x85K\x01\x85\x89ccollections\nOrderedDict\n)RtR"
    data = b"\x80\x04" + _pickled_text("n" * 1_000_000) + b"\x94"  # memo 0
    data += _pickled_text("t") + b"\x94" + tensor + b"\x94"  # memo 1, 2
    data += _script_object(b"h\x01h\x02s") + b"\x94"  # memo 3
    for i in range(6):
        fields = b"h\x01h\x02sh\x00h" + bytes([i + 3]) + b"s"
        data += _script_object(fields) + b"\x94"
    path = tmp_path / "model.pt"
    _torchscript_archive(path, data + b".", records=(("data/0", bytes(4)),))

    with pytest.raises(ValueError, match="TorchScript model files are not run"):
        ModelLoader().load_state_dict_from_file(path)


@pytest.mark.parametrize("extension", [".pt", ".pth", ".ckpt"])
@pytest.mark.parametrize("zipfile_format", [True, False])
def test_pickled_state_dict(tmp_path: Path, extension: str, zipfile_format: bool):
    expected = make_net().state_dict()
    path = tmp_path / f"model{extension}"
    torch.save(expected, path, _use_new_zipfile_serialization=zipfile_format)

    actual = ModelLoader().load_state_dict_from_file(path)

    assert_same_state_dict(actual, expected)


def test_pickled_state_dict_refuses_other_globals(tmp_path: Path):
    path = tmp_path / "model.pth"
    torch.save({"weight": torch.zeros(1), "loader": ModelLoader}, path)

    with pytest.raises(Exception, match="is forbidden"):
        ModelLoader().load_state_dict_from_file(path)


def _pickled_text(text: str) -> bytes:
    data = text.encode()
    return b"X" + struct.pack("<I", len(data)) + data


def _pickled_global(module: str, name: str) -> bytes:
    return f"c{module}\n{name}\n".encode()


@pytest.mark.parametrize("target", ["rebuild", "typing"])
def test_pickled_state_dict_cannot_change_later_loads(tmp_path: Path, target: str):
    # A pickle can set attributes on the globals it is given (BUILD). This one sets
    # the default metadata of the tensor rebuild function (negated tensors), or the
    # class behind typing's OrderedDict, for the rest of the process.
    if target == "rebuild":
        obj = _pickled_global("torch._utils", "_rebuild_tensor_v2")
        value = b"}" + _pickled_text("neg") + b"\x88s\x85"  # ({"neg": True},)
    else:
        obj = _pickled_global("typing", "OrderedDict")
        value = b"N"
    name = "__defaults__" if target == "rebuild" else "__origin__"
    # (None, {name: value}) BUILD on the global, then drop it and return {}
    payload = b"\x80\x02" + obj + b"N}" + _pickled_text(name) + value + b"s\x86b0}."
    path = tmp_path / "changes.pth"
    with path.open("wb") as file:
        header = {
            "protocol_version": 1001,
            "little_endian": True,
            "type_sizes": {"short": 2, "int": 4, "long": 4},
        }
        for value in (0x1950A86A20F9469CFC6C, 1001, header):
            pickle.dump(value, file, protocol=2)
        file.write(payload)
        pickle.dump([], file, protocol=2)
    expected = make_net().state_dict()
    plain = tmp_path / "model.pth"
    torch.save(expected, plain)

    if target == "rebuild":
        assert ModelLoader().load_state_dict_from_file(path) == {}
    else:
        with pytest.raises(TypeError, match="immutable type"):
            ModelLoader().load_state_dict_from_file(path)
    actual = ModelLoader().load_state_dict_from_file(plain)

    assert_same_state_dict(actual, expected)
    assert not any(value.is_neg() for value in actual.values())
    assert typing.OrderedDict() == collections.OrderedDict()


def test_safetensors(tmp_path: Path):
    expected = make_net().state_dict()
    path = tmp_path / "model.safetensors"
    save_file(expected, str(path))

    actual = ModelLoader().load_state_dict_from_file(path)

    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key
