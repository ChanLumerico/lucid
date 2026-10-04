"""One collector per tool, each turning that tool's findings into counts.

A collector is *local* when a file's count depends on that file alone
(ruff, the counters, lizard): the pre-commit fast path can then measure only
the changed files.  The cross-file ones (vulture, jscpd, mypy) need the
whole tree and run in ``--full`` and ``--diff``.
"""

import subprocess
from collections.abc import Mapping
from pathlib import Path

from tools.quality_gate.core import Counts, GateError, Scope


class Collector:
    name = ""
    local = False  # counts of a file depend on that file alone
    fast = False  # cheap enough for the pre-commit path
    default = True  # on unless asked for (False: stub / opt-in)
    in_diff = True  # part of `--diff` (land) without --full

    def version(self) -> str:
        return "?"

    def collect(self, scope: Scope) -> Counts:
        raise NotImplementedError


def run_tool(
    cmd: list[str],
    cwd: Path,
    ok_codes: tuple[int, ...] = (0,),
    env: Mapping[str, str] | None = None,
) -> str:
    """Run a tool; a crash or an unexpected exit code is a GateError, never zero findings."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=None if env is None else dict(env),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GateError(f"{cmd[0]}: not found ({exc})") from exc
    if proc.returncode not in ok_codes:
        tail = (proc.stderr or proc.stdout).strip()[-2000:]
        raise GateError(f"{' '.join(cmd[:4])}… exited {proc.returncode}:\n{tail}")
    return proc.stdout


def all_collectors() -> list[Collector]:
    from tools.quality_gate.collectors.clang_tidy import ClangTidyCollector
    from tools.quality_gate.collectors.complexity import LizardCollector
    from tools.quality_gate.collectors.counters import CountersCollector
    from tools.quality_gate.collectors.deadcode import VultureCollector
    from tools.quality_gate.collectors.duplication import JscpdCollector
    from tools.quality_gate.collectors.lint import RuffCollector
    from tools.quality_gate.collectors.semgrep_rules import SemgrepCollector
    from tools.quality_gate.collectors.typecheck import MypyCollector

    return [
        RuffCollector(),
        CountersCollector(),
        LizardCollector(),
        VultureCollector(),
        JscpdCollector(),
        MypyCollector(),
        SemgrepCollector(),
        ClangTidyCollector(),
    ]
