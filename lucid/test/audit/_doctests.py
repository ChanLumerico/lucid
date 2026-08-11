"""The documentation, checked against the code it documents.

A docstring example is a claim about behaviour, written by the author, in
the file the behaviour lives in.  Nothing was checking them: the package
has 5,499 doctest examples and **585 of them fail**, across 90 modules,
and not one of those failures had ever been reported by anything.

They are not all cosmetic.  Alongside the repr line-wrapping and the
float-precision drift there is ``_C_engine.Dtype.Float32``, an attribute
that does not exist, in the documented way to build a tensor from an
impl; and ``Tensor.is_contiguous`` promising ``False`` for a transpose in
an engine that materialises every view.  A user following the
documentation writes code that does not run.

This is a *floor*, not a target, for the same reason the line-coverage
stage is: 585 failures cannot be fixed in the change that first measures
them, and a gate that is red on arrival is a gate people learn to pass
with a flag.  The number is recorded, and the stage fails when it goes
**up** — a new docstring that does not run is caught on the commit that
adds it.

    lucid-audit                        # the doctest stage runs with the rest
    lucid-audit --update-doctests      # record the current count as the floor
    lucid-audit --no-doctests          # skip it
"""

import contextlib
import doctest
import importlib
import io
import json
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lucid

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Same exclusions as the sweep's surface walk, and for the same reasons:
#: the zoo has its own suites, ``lucid.test`` is this tool, ``_C`` is the
#: compiled engine and ``benchmarks`` are drivers.
_SKIP = ("lucid.models", "lucid.test", "lucid._C", "lucid.benchmarks")

#: ``NORMALIZE_WHITESPACE`` because a tensor repr wraps differently at
#: different widths and that is not a claim anyone meant to make.
#: ``ELLIPSIS`` because several docstrings already use ``...`` for a
#: shape or an address.
_OPTIONS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS


class DoctestResult:
    """What the documentation run found."""

    __slots__ = ("failed", "attempted", "per_module", "ran")

    def __init__(
        self,
        failed: int = 0,
        attempted: int = 0,
        per_module: "dict[str, int] | None" = None,
        ran: bool = False,
    ) -> None:
        self.failed = failed
        self.attempted = attempted
        self.per_module = per_module or {}
        self.ran = ran


def _modules() -> "Iterator[Any]":
    yield lucid
    for info in pkgutil.walk_packages(lucid.__path__, "lucid."):
        name = info.name
        if name.startswith(_SKIP):
            continue
        # Private packages stay: ``lucid._ops`` and ``lucid._tensor`` hold
        # most of the tensor API's documentation, and a docstring is
        # shipped whatever the module is called.
        try:
            yield importlib.import_module(name)
        except (
            Exception
        ):  # noqa: BLE001 - an unimportable module is the sweep's finding
            continue


def run() -> DoctestResult:
    """Every docstring example in the package, once."""
    failed = attempted = 0
    per_module: "dict[str, int]" = {}
    for module in _modules():
        buffer = io.StringIO()
        try:
            # Output captured: a failing example prints its own diff, and
            # 585 of those would bury the report this stage belongs to.
            # The count is the finding; ``--json`` carries the detail.
            with contextlib.redirect_stdout(buffer):
                outcome = doctest.testmod(
                    module,
                    verbose=False,
                    optionflags=_OPTIONS,
                    extraglobs={"lucid": lucid},
                )
        except Exception:  # noqa: BLE001 - surveying, not asserting
            continue
        failed += outcome.failed
        attempted += outcome.attempted
        if outcome.failed:
            per_module[module.__name__] = outcome.failed
    return DoctestResult(failed, attempted, per_module, ran=True)


def load_floor(path: Path) -> "dict[str, int] | None":
    """The recorded per-module failure counts, or ``None``."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except OSError, ValueError:
        return None
    floor = data.get("failing")
    return floor if isinstance(floor, dict) else None


def save_floor(path: Path, result: DoctestResult) -> None:
    payload = {
        "comment": [
            "Docstring examples that do not run, per module.",
            "",
            "A floor, not a target.  The stage fails when a module's count",
            "goes up, so a new example that does not run is caught on the",
            "commit that adds it; the standing 585 are a backlog, recorded",
            "here so they cannot grow quietly.",
            "",
            "Regenerate with:  lucid-audit --update-doctests",
        ],
        "failed": result.failed,
        "attempted": result.attempted,
        "failing": dict(sorted(result.per_module.items())),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def regressions(
    result: DoctestResult, floor: "dict[str, int]"
) -> "list[tuple[str, int, int]]":
    """``(module, was, now)`` for every module that got worse."""
    out: "list[tuple[str, int, int]]" = []
    for name, now in sorted(result.per_module.items()):
        was = floor.get(name, 0)
        if now > was:
            out.append((name, was, now))
    return out


__all__ = [
    "DoctestResult",
    "load_floor",
    "regressions",
    "run",
    "save_floor",
]
