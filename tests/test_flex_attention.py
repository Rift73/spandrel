import os
import subprocess
import sys
from pathlib import Path

SPANDREL = Path(__file__).parent.parent / "libs" / "spandrel"
ARCHS = (
    "spandrel.architectures.DRFT.__arch.drft_arch",
    "spandrel.architectures.DRFT.__arch.legacy_drft_arch",
    "spandrel.architectures.SST.__arch.sst_arch",
)


def _run(code: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(SPANDREL)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=240)


def test_loading_the_architectures_leaves_flex_attention_unloaded():
    names = ", ".join(repr(name) for name in ARCHS)
    _run(
        "import importlib, sys\n"
        f"for name in ({names}):\n"
        "    importlib.import_module(name)\n"
        "assert 'torch.nn.attention.flex_attention' not in sys.modules\n"
    )


def test_a_pytorch_without_attention_module_only_lacks_flex():
    names = ", ".join(repr(name) for name in ARCHS)
    _run(
        "import importlib, sys, torch\n"
        "sys.modules['torch.nn.attention'] = None\n"
        f"for name in ({names}):\n"
        "    assert importlib.import_module(name)._FLEX_AVAILABLE is False\n"
    )
