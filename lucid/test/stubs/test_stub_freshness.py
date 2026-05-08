"""Type-stub freshness regression.

Ensures the checked-in ``.pyi`` files match what ``tools/gen_pyi.py``
would emit right now.  Catches drift between the ops registry and the
stubs that downstream type-checkers consume.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GEN_PYI = _REPO_ROOT / "tools" / "gen_pyi.py"
_CHECK_STUBS = _REPO_ROOT / "tools" / "check_stubs.py"


@pytest.mark.smoke
def test_check_stubs_passes() -> None:
    """``tools/check_stubs.py`` exits cleanly when stubs are fresh."""
    if not _CHECK_STUBS.exists():
        pytest.skip(f"{_CHECK_STUBS} not present")
    result = subprocess.run(
        [sys.executable, str(_CHECK_STUBS)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"check_stubs.py failed:\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


@pytest.mark.smoke
def test_gen_pyi_script_present() -> None:
    """The stub-generation script itself is on disk."""
    assert _GEN_PYI.exists(), f"{_GEN_PYI} missing"
