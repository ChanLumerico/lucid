"""vulture: code nothing reaches.

The whole of ``lucid/`` is scanned, tests included, because the tests are the
public API's users — without them every public method nothing inside the
library calls would read as dead.  Only findings in production files count.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector, run_tool
from tools.quality_gate.core import Counts, Scope, is_py_target

_LINE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+): (?P<msg>.*?) \(\d+% confidence")
WHITELIST = Path(__file__).resolve().parents[1] / "vulture_whitelist.py"


def _rule(message: str) -> str:
    head = message.split("'", 1)[0].strip()
    return re.sub(r"[^a-z]+", "-", head.lower()).strip("-") or "unknown"


class VultureCollector(Collector):
    name = "vulture"

    @override
    def version(self) -> str:
        return run_tool([sys.executable, "-m", "vulture", "--version"], Path.cwd()).split()[-1]

    @override
    def collect(self, scope: Scope) -> Counts:
        cmd = [
            sys.executable,
            "-m",
            "vulture",
            config.PY_ROOT.rstrip("/"),
            str(WHITELIST),
            f"--min-confidence={config.VULTURE_MIN_CONFIDENCE}",
            "--ignore-names=" + ",".join(config.VULTURE_IGNORE_NAMES),
        ]
        # 0: nothing found, 3: dead code found.  1/2 are input/usage errors.
        out = run_tool(cmd, scope.root, ok_codes=(0, 3))
        counts: Counts = defaultdict(int)
        for line in out.splitlines():
            m = _LINE.match(line)
            if m and is_py_target(m["path"]):
                counts[("vulture", _rule(m["msg"]), m["path"])] += 1
        return dict(counts)
