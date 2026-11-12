import os
import subprocess
import sys
from pathlib import Path

SPANDREL = Path(__file__).parent.parent / "libs" / "spandrel"


def test_importing_spandrel_leaves_torchvision_unloaded():
    env = {**os.environ, "PYTHONPATH": str(SPANDREL)}
    code = (
        "import sys\n"
        "import spandrel\n"
        "from spandrel.architectures.LaMa import LaMaArch\n"
        "from spandrel.architectures.RestoreFormer import RestoreFormerArch\n"
        "assert 'torchvision' not in sys.modules, 'torchvision was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=240)
