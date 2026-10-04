"""ruff with the deslop rule set on top of the project's own lint config."""

import functools
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector, run_tool
from tools.quality_gate.core import Counts, GateError, Scope


@functools.cache
def ruff_bin() -> str:
    """The binary of the ruff *distribution* installed with this interpreter.

    Not ``python -m ruff``: from a worktree's light venv that resolves to
    whatever ``ruff`` is first on PATH (0.15 there, 0.16 in the shared venv on
    2026-10-05), and a different ruff counts differently.
    """
    try:
        dist = importlib.metadata.distribution("ruff")
    except importlib.metadata.PackageNotFoundError as exc:
        raise GateError("ruff is not installed (pip install --group quality)") from exc
    for f in dist.files or ():
        if f.name == "ruff" and "bin" in f.parts:
            path = Path(str(dist.locate_file(f))).resolve()
            if path.is_file():
                return str(path)
    raise GateError("ruff's distribution ships no bin/ruff")


class RuffCollector(Collector):
    name = "ruff"
    local = True
    fast = True

    @override
    def version(self) -> str:
        return run_tool([ruff_bin(), "--version"], Path.cwd()).split()[-1]

    @override
    def collect(self, scope: Scope) -> Counts:
        if not scope.py:
            return {}
        out = run_tool(
            [
                ruff_bin(),
                "check",
                "--no-cache",
                "--exit-zero",
                "--output-format=json",
                f"--extend-select={config.RUFF_EXTEND_SELECT}",
                f"--extend-ignore={config.RUFF_IGNORE}",
                *scope.py,
            ],
            scope.root,
        )
        try:
            findings = json.loads(out or "[]")
        except json.JSONDecodeError as exc:
            raise GateError(f"ruff: unreadable output ({exc})") from exc
        counts: Counts = defaultdict(int)
        root = scope.root.resolve()
        for f in findings:
            code = f.get("code") or "syntax-error"
            path = Path(f["filename"]).resolve().relative_to(root).as_posix()
            counts[("ruff", code, path)] += 1
        return dict(counts)
