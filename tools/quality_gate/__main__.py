"""python -m tools.quality_gate — see tools/README.md "Quality gate"."""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from tools.quality_gate import gate
from tools.quality_gate.collectors import Collector, all_collectors
from tools.quality_gate.core import (
    GateError,
    MergeRefused,
    baseline_to_json,
    changed_files,
    load_baseline,
    merge_baselines,
)


def _pick(names: str | None, *, fast: bool, diff: bool, full: bool) -> list[Collector]:
    every = all_collectors()
    if names:
        wanted = {n.strip() for n in names.split(",") if n.strip()}
        unknown = wanted - {c.name for c in every}
        if unknown:
            raise GateError(f"unknown collector(s): {', '.join(sorted(unknown))}")
        chosen = [c for c in every if c.name in wanted]
    elif fast:
        chosen = [c for c in every if c.fast]
    elif diff and not full:
        chosen = [c for c in every if c.default and c.in_diff]
    else:
        chosen = [c for c in every if c.default]
    if fast and any(not c.local for c in chosen):
        raise GateError("--fast measures changed files only; cross-file collectors need --full")
    return chosen


def _merge_driver(paths: list[str]) -> int:
    """git merge driver: ``%O %A %B`` — the result goes to %A."""
    base, ours, theirs = (Path(p) for p in paths)
    try:
        merged = merge_baselines(load_baseline(base), load_baseline(ours), load_baseline(theirs))
    except (MergeRefused, GateError) as exc:
        print(f"[quality_gate] baseline merge refused: {exc}", file=sys.stderr)
        return 1
    ours.write_text(baseline_to_json(merged))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tools.quality_gate",
        description="Ratchet gate: no (tool, rule, file) count may rise above the baseline.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="changed files, cheap collectors")
    mode.add_argument("--full", action="store_true", help="whole tree, every collector")
    p.add_argument("--diff", metavar="REF", help="print the SLOP DELTA against REF")
    p.add_argument("--report", action="store_true", help="totals per defect class")
    p.add_argument("--update", action="store_true", help="lower the baseline (never raises)")
    p.add_argument("--rebaseline", action="store_true", help="rewrite it (orchestrator only)")
    p.add_argument("--reason", default="", help="why, for --rebaseline (recorded)")
    p.add_argument("--collectors", help="comma-separated subset, e.g. ruff,counters")
    p.add_argument("--top", type=int, default=25, help="rules listed by --report")
    p.add_argument("--root", default=".", help="repository root (default: cwd)")
    p.add_argument("--merge-baseline", nargs=3, metavar=("O", "A", "B"), help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args.merge_baseline:
        return _merge_driver(args.merge_baseline)
    if sum(map(bool, (args.diff, args.report, args.update, args.rebaseline))) > 1:
        p.error("choose one of --diff / --report / --update / --rebaseline")
    if args.rebaseline and args.fast:
        p.error("--rebaseline measures the whole tree; drop --fast")

    workdir = Path(tempfile.mkdtemp(prefix="quality-gate-"))
    try:
        return _run(args, Path(args.root).resolve(), workdir)
    except GateError as exc:
        print(f"[quality_gate] cannot measure: {exc}", file=sys.stderr)
        return 2
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run(args: argparse.Namespace, root: Path, workdir: Path) -> int:
    collectors = _pick(args.collectors, fast=args.fast, diff=bool(args.diff), full=args.full)
    files = changed_files(root, "HEAD") if args.fast else None
    if args.diff:
        return gate.diff(root, collectors, args.diff, workdir)
    if args.report:
        return gate.report(root, collectors, workdir, args.top)
    if args.update:
        return gate.update(root, collectors, files, workdir)
    if args.rebaseline:
        print("⚠️  --rebaseline may RAISE counts: orchestrator only.", file=sys.stderr)
        return gate.rebaseline(root, collectors, args.reason, workdir)
    return gate.check(root, collectors, files, workdir)


if __name__ == "__main__":
    sys.exit(main())
