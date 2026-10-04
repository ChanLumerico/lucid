"""ruff with the deslop rule set on top of the project's own lint config."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector, run_tool
from tools.quality_gate.core import Counts, GateError, Scope


class RuffCollector(Collector):
    name = "ruff"
    local = True
    fast = True

    @override
    def version(self) -> str:
        return run_tool([sys.executable, "-m", "ruff", "--version"], Path.cwd()).split()[-1]

    @override
    def collect(self, scope: Scope) -> Counts:
        if not scope.py:
            return {}
        out = run_tool(
            [
                sys.executable,
                "-m",
                "ruff",
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
