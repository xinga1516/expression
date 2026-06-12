from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script",
    [
        "scripts/train.py",
        "scripts/annotate_emtab_cells.py",
        "scripts/evaluate.py",
        "scripts/model_test.py",
        "scripts/pretrain_scvi.py",
        "scripts/project_test.py",
    ],
)
def test_cli_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
