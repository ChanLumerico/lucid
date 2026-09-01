"""The generated Core ML field numbers still match the specification.

``lucid/_C/coreml/MilSchema.h`` is committed and compiled into the engine,
so nothing at runtime needs coremltools.  That also means a Core ML
specification bump can silently invalidate it: a changed field number
produces a well-formed message whose reader ignores or misreads the field,
and every value test would keep passing.

This is the guard.  It runs only where the reference descriptors are
installed, which is the same opt-in standing the reference framework has
for parity tests.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "coremltools", reason="schema drift is checked against the reference descriptors"
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_the_committed_schema_matches_the_descriptors() -> None:
    result = subprocess.run(
        [sys.executable, "tools/gen_mil_schema.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "lucid/_C/coreml/MilSchema.h is out of date with the installed Core ML "
        f"descriptors — run `python tools/gen_mil_schema.py`.\n{result.stderr}"
    )
