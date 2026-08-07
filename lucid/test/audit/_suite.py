"""The test suite as a stage of the audit rather than a separate errand.

The audit walks the package and asks whether each reachable symbol keeps
its contract.  The suite asks whether specific values are the right
values.  They fail independently — and this session's evidence is that
most defects are only visible to one of them:

* the sweep found the gradient that was never wired and the sampler
  drawing at the wrong concentration;
* the suite found the assignment writing a rectangle instead of a
  diagonal, the histogram binning onto the wrong grid, the transform
  whose inverse was NaN, and the window that deleted its own frame.

So a gate that runs only one of them reports "clean" while half the
framework is unchecked.  This module runs the other half and folds its
result into the same verdict, which is what makes ``lucid-audit`` a
single command rather than the first of two.

Line coverage rides along because it costs nothing: the suite takes
9m09s uninstrumented and 8m29s under ``coverage`` — the wall clock is
MLX and Accelerate, not Python line tracing.  A floor that is free to
check should be checked on every run rather than remembered.
"""

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.test.audit._console import Console


#: How far a module's line coverage may fall before it is called a
#: regression.  Not zero: adding statements to a well-covered file lowers
#: its percentage honestly, and a gate that fires on that teaches people
#: to ignore it.
_FILE_TOLERANCE = 2.0

#: The same, for the total.  Tighter, because the total absorbs the noise
#: of individual files and a real drop shows up here first.
_TOTAL_TOLERANCE = 0.10

#: Files below this many statements are not worth a per-file verdict —
#: one uncovered line in a twelve-line module is a 8% swing.
_MIN_STATEMENTS = 20

_COUNT = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)")
_FAILURE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)")


@dataclass
class SuiteResult:
    """What one run of the suite established."""

    ran: bool = False
    returncode: int = 0
    duration: float = 0.0
    counts: "dict[str, int]" = field(default_factory=dict)
    failures: "list[str]" = field(default_factory=list)
    line_covered: "int | None" = None
    line_total: "int | None" = None
    per_file: "dict[str, tuple[int, int]]" = field(default_factory=dict)
    unavailable: "str | None" = None

    @property
    def percent(self) -> "float | None":
        """Line coverage as a percentage, or ``None`` if not measured."""
        if self.line_total in (None, 0) or self.line_covered is None:
            return None
        assert self.line_total is not None
        return 100.0 * self.line_covered / self.line_total

    @property
    def broken(self) -> int:
        """Failures plus errors — the count that makes the verdict red."""
        return self.counts.get("failed", 0) + self.counts.get(
            "error", self.counts.get("errors", 0)
        )


def run_suite(
    console: "Console",
    path: str = "lucid/test",
    *,
    with_coverage: bool = True,
    root: "Path | None" = None,
) -> SuiteResult:
    """Run the suite in a subprocess and return what it established.

    A subprocess rather than ``pytest.main`` in-process: the audit has
    already imported the whole package and, on some axes, patched parts of
    it.  Collecting the suite into that interpreter would let one stage's
    state decide the other stage's result, which is the one thing a gate
    made of two independent checks must not allow.

    Parameters
    ----------
    console : Console
        Where progress is echoed while the child runs.
    path : str, optional
        What to hand pytest.  Default ``"lucid/test"`` — the whole tree.
        ``pytest.ini_options`` already deselects the audit sweep (marked
        ``audit``) and the heavy models, so this is "everything except
        the stage that is running right now".
    with_coverage : bool, optional
        Measure line coverage alongside.  Default ``True``.
    root : Path, optional
        Working directory.  Defaults to the repository root inferred from
        this file.

    Returns
    -------
    SuiteResult
        ``ran=False`` with ``unavailable`` set when the suite could not be
        started at all — a wheel install, or ``coverage`` missing.
    """
    root = root or Path(__file__).resolve().parents[3]
    if not (root / path).exists():
        return SuiteResult(unavailable=f"{path} is not in this checkout")

    argv = [sys.executable]
    coverage_file = root / ".coverage.audit"
    if with_coverage:
        try:
            import coverage  # noqa: F401
        except ImportError:
            with_coverage = False
            console.always(
                console.paint(
                    "  coverage is not installed — running without the line floor",
                    "grey",
                )
            )
    if with_coverage:
        argv += ["-m", "coverage", "run", f"--data-file={coverage_file}"]
    argv += [
        "-m",
        "pytest",
        path,
        "-q",
        "-rfE",
        "--no-header",
        "-p",
        "no:cacheprovider",
        # The child colours its own output, so it has to be told what the
        # parent decided — otherwise ``--no-color`` produces a transcript
        # full of escape sequences, which is exactly what that flag is for.
        f"--color={'yes' if console.colour else 'no'}",
    ]

    console.always(console.paint(f"  running {path} — this is the long stage", "grey"))

    started = time.time()
    proc = subprocess.Popen(
        argv,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    result = SuiteResult(ran=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        match = _FAILURE.match(line)
        if match:
            result.failures.append(match.group(1))
        for count, label in _COUNT.findall(line):
            if label.startswith("error"):
                label = "error"
            result.counts[label] = int(count)
        if line.strip():
            # ``write`` rather than ``always``: under ``--quiet`` the child's
            # chatter is exactly what should disappear, and the counts and
            # failures below survive because they go through ``always``.
            console.write(console.paint(f"  │ {line[:160]}", "grey"))
    result.returncode = proc.wait()
    result.duration = time.time() - started

    if with_coverage:
        _read_coverage(result, root, coverage_file, console)
    return result


def _read_coverage(
    result: SuiteResult, root: Path, data_file: Path, console: "Console"
) -> None:
    """Turn the collected data into per-file covered/total counts."""
    out = root / ".coverage.audit.json"
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                f"--data-file={data_file}",
                "-o",
                str(out),
                "-q",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
        payload = json.loads(out.read_text())
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as exc:
        console.always(
            console.paint(f"  could not read the coverage data: {exc!r}", "grey")
        )
        return
    finally:
        for leftover in (out, data_file):
            leftover.unlink(missing_ok=True)

    totals = payload["totals"]
    result.line_covered = totals["covered_lines"]
    result.line_total = totals["num_statements"]
    result.per_file = {
        name: (entry["summary"]["covered_lines"], entry["summary"]["num_statements"])
        for name, entry in payload["files"].items()
    }


# ── the recorded floor ───────────────────────────────────────────────────────


def load_floor(path: Path) -> "dict[str, object] | None":
    """Read the recorded line-coverage floor, or ``None`` when absent."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def save_floor(path: Path, result: SuiteResult) -> None:
    """Record this run as the floor later runs are measured against."""
    payload = {
        "line_covered": result.line_covered,
        "line_total": result.line_total,
        "percent": result.percent,
        "files": {
            name: {"covered": covered, "total": total}
            for name, (covered, total) in sorted(result.per_file.items())
        },
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def report_line_coverage_diff(
    result: SuiteResult, floor: "dict[str, object]", console: "Console"
) -> int:
    """Compare against the floor and report what moved.

    Percentages rather than raw counts, because both numerator and
    denominator move legitimately: deleting dead code lowers the covered
    count without lowering coverage, and adding a well-tested module
    raises both.  A tolerance rather than equality, because a file that
    grows by a statement drops a fraction of a point honestly and a gate
    that fires on that gets muted.

    Returns
    -------
    int
        How many regressions were found.  Non-zero makes the run red.
    """
    now = result.percent
    was = floor.get("percent")
    if now is None or not isinstance(was, (int, float)):
        return 0

    recorded = floor.get("files")
    lost: list[tuple[str, float, float]] = []
    if isinstance(recorded, dict):
        for name, entry in recorded.items():
            if not isinstance(entry, dict):
                continue
            old_total = entry.get("total", 0)
            if not isinstance(old_total, int) or old_total < _MIN_STATEMENTS:
                continue
            old_covered = entry.get("covered", 0)
            assert isinstance(old_covered, int)
            before = 100.0 * old_covered / old_total
            current = result.per_file.get(name)
            if current is None:
                continue  # moved or deleted — not a coverage regression
            covered, total = current
            after = 100.0 * covered / total if total else 100.0
            if before - after > _FILE_TOLERANCE:
                lost.append((name, before, after))

    if not lost and now >= was - _TOTAL_TOLERANCE:
        console.always(
            console.paint(
                f"  line coverage {now:.2f}% — at or above the recorded " f"{was:.2f}%",
                "green",
            )
        )
        if now > was + _TOTAL_TOLERANCE:
            console.always(
                console.paint(
                    "  run with --update-suite to record the new floor", "grey"
                )
            )
        return 0

    console.rule(f"line coverage regressions · {len(lost) or 1}", "red")
    if now < was - _TOTAL_TOLERANCE:
        console.always(console.paint(f"  TOTAL  {was:.2f}% → {now:.2f}%", "red"))
    for name, before, after in sorted(lost, key=lambda r: r[1] - r[2], reverse=True):
        console.always(
            console.paint(f"  LOST   {name}  {before:.1f}% → {after:.1f}%", "red")
        )
    console.always(
        console.paint(
            "  a module that stopped being exercised is what a refactor "
            "breaks without breaking a test",
            "grey",
        )
    )
    console.always(
        console.paint("  run with --update-suite to accept the new floor", "grey")
    )
    return len(lost) or 1


def report_suite(result: SuiteResult, console: "Console") -> None:
    """Print what the suite established, defects first."""
    if not result.ran:
        console.always(console.paint(f"  suite not run: {result.unavailable}", "grey"))
        return

    counts = result.counts
    parts = [f"{counts.get('passed', 0)} passed"]
    if result.broken:
        parts.append(f"{result.broken} failed")
    for label in ("skipped", "xfailed", "xpassed"):
        if counts.get(label):
            parts.append(f"{counts[label]} {label}")

    colour = "red" if result.broken else "green"
    console.always("")
    console.always(
        console.paint("  suite".ljust(22), "grey")
        + console.paint(", ".join(parts), colour)
        + console.paint(f"   {result.duration:.0f}s", "grey")
    )
    if result.percent is not None:
        console.always(
            console.paint("  line coverage".ljust(22), "grey")
            + console.paint(
                f"{result.percent:.2f}%  "
                f"({result.line_covered}/{result.line_total})",
                "cyan",
            )
        )

    if result.failures:
        console.rule(f"suite failures · {len(result.failures)}", "red")
        for nodeid in result.failures[:40]:
            console.always(console.paint(f"  FAIL  {nodeid}", "red"))
        if len(result.failures) > 40:
            console.always(
                console.paint(f"  … and {len(result.failures) - 40} more", "grey")
            )
    elif result.broken:
        # Counts said something broke but no nodeid was parsed — say so
        # rather than printing a clean list and implying there is none.
        console.always(
            console.paint(
                f"  {result.broken} failure(s) reported without a parseable "
                "node id — run pytest directly",
                "red",
            )
        )


__all__ = [
    "SuiteResult",
    "load_floor",
    "report_line_coverage_diff",
    "report_suite",
    "run_suite",
    "save_floor",
]
