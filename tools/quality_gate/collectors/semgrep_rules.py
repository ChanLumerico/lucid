"""semgrep with Lucid's own rules from ``tools/semgrep/rules/``.

The framework only: the rules arrive with DS-4 (LCD-263).  Until a rule file
exists the collector measures nothing and costs nothing; once one exists it
runs in ``--full`` and counts per rule id.
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector, run_tool
from tools.quality_gate.core import Counts, GateError, Scope, is_cpp_target, is_py_target

RULES = Path(__file__).resolve().parents[2] / "semgrep" / "rules"


def rule_files() -> list[Path]:
    return sorted(p for p in RULES.glob("*.y*ml")) if RULES.is_dir() else []


class SemgrepCollector(Collector):
    name = "semgrep"
    in_diff = False

    @override
    def version(self) -> str:
        return "none" if not rule_files() else self._run(["--version"], Path.cwd()).strip()

    def _run(self, args: list[str], cwd: Path) -> str:
        exe = shutil.which("semgrep") or str(Path(sys.executable).parent / "semgrep")
        return run_tool([exe, *args], cwd, ok_codes=(0, 1))

    @override
    def collect(self, scope: Scope) -> Counts:
        rules = rule_files()
        if not rules:
            return {}
        args = ["scan", "--json", "--metrics=off", "--disable-version-check", "--quiet"]
        args += [f"--config={rule}" for rule in rules]
        out = self._run([*args, config.PY_ROOT.rstrip("/")], scope.root)
        try:
            results = json.loads(out)["results"]
        except (KeyError, json.JSONDecodeError) as exc:
            raise GateError(f"semgrep: unreadable output ({exc})") from exc
        counts: Counts = defaultdict(int)
        for r in results:
            path = r["path"]
            if is_py_target(path) or is_cpp_target(path):
                counts[("semgrep", r["check_id"].rsplit(".", 1)[-1], path)] += 1
        return dict(counts)
