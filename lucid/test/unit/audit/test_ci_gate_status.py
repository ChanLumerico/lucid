"""A broken instrument cannot turn a required CI stage green."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "stage,next_marker",
    [
        ("Symbol x axis audit", "# Docstring examples"),
        (
            "Docstring examples (against the doctest floor)",
            "# Model-zoo family contract",
        ),
    ],
)
@pytest.mark.parametrize("status", [0, 1, 2])
def test_required_ci_stage_preserves_failure(
    tmp_path: Path,
    stage: str,
    next_marker: str,
    status: int,
) -> None:
    root = Path(__file__).resolve().parents[4]
    script = (root / "scripts/ci_full.sh").read_text()
    block = script.split(f'echo "==> {stage}"', 1)[1].split(next_marker, 1)[0]
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport os, sys\nsys.exit(int(os.environ['TEST_STAGE_STATUS']))\n"
    )
    fake_python.chmod(0o700)
    result = subprocess.run(
        ["bash", "-c", block + "\nprintf 'NEXT_STAGE'"],
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "TEST_STAGE_STATUS": str(status),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stdout + result.stderr
    assert ("NEXT_STAGE" in result.stdout) == (status == 0)
