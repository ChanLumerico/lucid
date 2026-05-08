"""Golden-tensor I/O.

Numerical / stability tests that depend on a fixed reference (computed
once, then frozen) load their expected output from
``numerical/golden/*.npz``.  This helper standardises the path
resolution and the on-disk format.

A golden file is just a NumPy ``.npz`` archive — keys map to expected
arrays.  Tests that detect a missing key surface ``pytest.skip`` so
new ops don't have to gate the suite while a baseline is being
established.
"""

from pathlib import Path

import numpy as np
import pytest


_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "numerical" / "golden"


def golden_path(name: str) -> Path:
    return _GOLDEN_DIR / f"{name}.npz"


def load_golden(name: str, key: str) -> np.ndarray:
    path = golden_path(name)
    if not path.exists():
        pytest.skip(f"golden file missing: {path.name}")
    arch = np.load(path)
    if key not in arch.files:
        pytest.skip(f"golden key missing: {name}.npz#{key}")
    return arch[key]


def save_golden(name: str, **arrays: np.ndarray) -> None:
    """Write or overwrite ``numerical/golden/<name>.npz``.

    Used during a deliberate "rebaseline" workflow — never called from
    inside the test suite itself.  Keep the file small (target ≤ 64
    KB per archive) so the repo stays light.
    """
    _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(golden_path(name), **arrays)
