"""Run test files one process each, with nothing filtered and nothing skipped.

The nightly suites exist so that no test goes unrun.  Three things used to let
one slip: a marker the default ``addopts`` deselects (``heavy``, ``control``),
an environment switch nobody set (``LUCID_TEST_NETWORK``), and a skip — an
oracle not installed, an API that drifted, a combination that does not apply.
This runs every given file with the markers cleared and the network on, and
treats a skip as a failure: a test that cannot run here should not exist, or
should assert what it can instead.

One child process per file, because a single interpreter carrying the model
zoo or the parity tier from file to file outgrows a hosted runner's memory.

usage: python scripts/ci_nightly_suite.py PATH [PATH ...] [--exclude PATH ...]
       (a directory is expanded to its ``test_*.py`` files; ``--exclude``
       drops files so another job can run them — ci.yml gives the long
       training parity its own)
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SUMMARY = re.compile(r"^(?:=+ )?(\d+ .*?) in [0-9.]+s")
_TIMEOUT = 3600


def _files(args: list[str]) -> list[Path]:
    found: list[Path] = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            found.extend(sorted(path.rglob("test_*.py")))
        else:
            found.append(path)
    return found


def _run(path: Path) -> tuple[str, list[str], float]:
    """Run one file; return (verdict, detail lines, seconds)."""
    env = dict(os.environ, LUCID_TEST_NETWORK="1")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(path),
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-rsfE",
        "--tb=short",
        "--color=no",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT, env=env
        )
    except subprocess.TimeoutExpired:
        return "timeout", [f"exceeded {_TIMEOUT}s"], time.time() - t0
    secs = time.time() - t0
    lines = proc.stdout.splitlines()
    skips = [ln for ln in lines if ln.startswith("SKIPPED")]
    fails = [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))]
    summary = next(
        (m.group(1) for ln in reversed(lines) if (m := _SUMMARY.search(ln))), ""
    )
    if proc.returncode not in (0, 5) or fails:
        detail = fails or lines[-25:]
        return "fail", [summary, *detail], secs
    if skips or proc.returncode == 5:
        return "skip", [summary, *skips] if skips else [summary, "no test ran"], secs
    return "ok", [summary], secs


def main() -> int:
    args = sys.argv[1:]
    excluded: set[Path] = set()
    if "--exclude" in args:
        cut = args.index("--exclude")
        args, excluded = args[:cut], {p.resolve() for p in _files(args[cut + 1 :])}
    files = [p for p in _files(args) if p.resolve() not in excluded]
    if not files:
        # An empty selection reporting "0 not clean" would pass a job that
        # ran nothing.
        print("no test files selected", file=sys.stderr)
        return 1
    bad: list[tuple[Path, str, list[str]]] = []
    started = time.time()
    flaky: list[Path] = []
    for path in files:
        verdict, detail, secs = _run(path)
        if verdict in ("fail", "timeout"):
            # Once more before calling it: a first download of a large
            # checkpoint can stall past the reader's timeout and pass on the
            # next try.  Listed as flaky either way, never hidden.
            verdict2, detail2, secs2 = _run(path)
            if verdict2 == "ok":
                flaky.append(path)
                verdict, detail, secs = "flaky", [detail2[0], *detail], secs + secs2
            else:
                verdict, detail, secs = verdict2, detail2, secs + secs2
        print(
            f"{verdict:7s} {secs:7.1f}s {path} — {detail[0] if detail else ''}",
            flush=True,
        )
        if verdict not in ("ok", "flaky"):
            bad.append((path, verdict, detail))
    print(
        f"\n{len(files)} files in {(time.time() - started) / 60:.1f} min, {len(bad)} not clean"
    )
    for path in flaky:
        print(f"FLAKY (failed, then passed on retry): {path}")
    for path, verdict, detail in bad:
        print(f"\n{verdict.upper()}: {path}")
        for line in detail[1:]:
            print(f"    {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
