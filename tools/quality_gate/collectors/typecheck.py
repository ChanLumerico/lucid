"""mypy with the suppression-auditing error codes, counted per code.

``mypy --strict lucid/`` is already a zero-error gate; what this adds is the
codes that find *laundering* — an ignore that no longer suppresses anything,
a cast to the type the value already has, a condition that is always true.
Slow next to the others (~25 s warm), so it runs in ``--full`` only.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector, run_tool
from tools.quality_gate.core import Counts, GateError, Scope, is_py_target


class MypyCollector(Collector):
    name = "mypy"
    in_diff = False

    @override
    def version(self) -> str:
        return run_tool([sys.executable, "-m", "mypy", "--version"], Path.cwd()).split()[1]

    @override
    def collect(self, scope: Scope) -> Counts:
        out = run_tool(
            [
                sys.executable,
                "-m",
                "mypy",
                config.PY_ROOT.rstrip("/"),
                *config.MYPY_FLAGS,
                "--output=json",
                "--no-error-summary",
            ],
            scope.root,
            ok_codes=(0, 1),  # 2 is a crash or a usage error
        )
        counts: Counts = defaultdict(int)
        for line in out.splitlines():
            if not line.strip():
                continue
            try:
                finding = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"mypy: unreadable line {line[:200]!r}") from exc
            if finding.get("severity") != "error" or not is_py_target(finding["file"]):
                continue
            counts[("mypy", finding.get("code") or "misc", finding["file"])] += 1
        return dict(counts)
