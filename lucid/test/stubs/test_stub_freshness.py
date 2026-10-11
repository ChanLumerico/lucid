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


_NS_STUB = '''
class _demo_namespace:
    @staticmethod
    def kept(a: int, b: int = ...) -> int: ...
    @staticmethod
    def renamed(x: int) -> int: ...
    @staticmethod
    def gone(a: int) -> int: ...
    LIMIT: int

demo: _demo_namespace
'''


def _bound(signature: str) -> object:
    """A stand-in for a pybind11 function: only its first docstring line is read."""

    def fn() -> None: ...

    fn.__doc__ = signature
    return fn


def test_namespace_check_reports_every_kind_of_drift() -> None:
    """LCD-402: hand-written sub-namespace stubs are compared with the engine —
    a stub-only name, an engine-only name, and a parameter list that differs
    in names or arity each fail, while a matching member passes."""
    import types

    from tools.stub_namespaces import namespace_drift

    demo = types.SimpleNamespace(
        kept=_bound("kept(a: int, b: int = 1) -> int"),
        renamed=_bound("renamed(input: int, flag: bool = False) -> int"),
        extra=_bound("extra(a: int) -> int"),
        LIMIT=3,
    )
    engine = types.SimpleNamespace(demo=demo)
    problems = namespace_drift(_NS_STUB, engine)
    assert problems == [
        "demo.gone: in the stub, not in the engine",
        "demo.extra: in the engine, not in the stub",
        "demo.renamed: stub (x), engine (input, flag)",
    ]


_CLASS_STUB = """
class Demo:
    def kept(self, a: int, b: int = ...) -> int: ...
    def renamed(self, x: int) -> int: ...
    def forwarded(self, value: int) -> int: ...
    def gone(self) -> None: ...
    @overload
    def size(self, dim: int) -> int: ...
    @overload
    def size(self, axis: None = ...) -> tuple[int, ...]: ...
    @property
    def width(self) -> int: ...
"""


class Demo:
    def kept(self, a: int, b: int = 1) -> int:
        return a + b

    def renamed(self, input: int) -> int:
        return input

    def forwarded(self, *args: object, **kwargs: object) -> int:
        return 0

    def size(self, dim: int | None = None) -> int:
        return 0

    @property
    def width(self) -> int:
        return 0

    def missing(self) -> None:
        return None


def test_class_check_reports_every_kind_of_drift() -> None:
    """LCD-357: a stub class is held to its runtime class — a stub-only
    name, a runtime-only name, a renamed parameter, a forwarding method
    whose callee binds other names, and an overload naming a parameter the
    runtime lacks each fail; matching members and properties pass."""
    from tools.stub_namespaces import class_drift

    problems = class_drift(
        _CLASS_STUB, Demo, lambda name: ["mask", "value"] if name == "forwarded" else None
    )
    assert problems == [
        "Demo.gone: in the stub, not at runtime",
        "Demo.missing: at runtime, not in the stub",
        "Demo.forwarded: stub (value), runtime (mask, value)",
        "Demo.renamed: stub (x), runtime (input)",
        "Demo.size: overload names (axis), runtime (dim)",
    ]
