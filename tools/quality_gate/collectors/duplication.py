"""jscpd: copy-pasted blocks across Python and C++.

A file's count is the number of clone pairs it takes part in (a pair inside
one file counts once).  jscpd runs from npx at a pinned version; without
Node the collector fails rather than reporting no duplication.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector, run_tool
from tools.quality_gate.core import Counts, GateError, Scope, is_cpp_target, is_py_target


def _npx() -> str:
    found = shutil.which("npx") or (
        "/opt/homebrew/bin/npx" if os.path.exists("/opt/homebrew/bin/npx") else None
    )
    if found is None:
        raise GateError("jscpd needs Node's npx on PATH (brew install node)")
    return found


class JscpdCollector(Collector):
    name = "jscpd"

    @override
    def version(self) -> str:
        return config.JSCPD_VERSION

    @override
    def collect(self, scope: Scope) -> Counts:
        out_dir = scope.workdir / "jscpd"
        out_dir.mkdir(parents=True, exist_ok=True)
        npx = _npx()
        # npx's shebang finds `node` through PATH: put npx's own directory first.
        path = f"{Path(npx).parent}{os.pathsep}{os.environ.get('PATH', '')}"
        env = {**os.environ, "PATH": path}
        run_tool(
            [
                npx,
                "-y",
                f"jscpd@{config.JSCPD_VERSION}",
                config.PY_ROOT.rstrip("/"),
                f"--format={config.JSCPD_FORMATS}",
                "--ignore=lucid/test/**,**/*.pyi",
                f"--min-lines={config.JSCPD_MIN_LINES}",
                f"--min-tokens={config.JSCPD_MIN_TOKENS}",
                "--max-size=5mb",
                "--reporters=json",
                f"--output={out_dir}",
            ],
            scope.root,
            env=env,
        )
        report = out_dir / "jscpd-report.json"
        try:
            duplicates = json.loads(report.read_text())["duplicates"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise GateError(f"jscpd: no readable report ({exc})") from exc
        counts: Counts = defaultdict(int)
        for dup in duplicates:
            files = set()
            for side in ("firstFile", "secondFile"):
                name = dup[side]["name"]
                path = name if name.startswith(config.PY_ROOT) else config.PY_ROOT + name
                if is_py_target(path) or is_cpp_target(path):
                    files.add(path)
            for path in files:
                counts[("jscpd", "clone-pair", path)] += 1
        return dict(counts)
