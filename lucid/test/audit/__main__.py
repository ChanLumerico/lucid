"""``python -m lucid.test.audit`` — the exhaustive sweep.

Examples
--------
::

    python -m lucid.test.audit                       # everything
    python -m lucid.test.audit --quick               # fewer domains, smaller probes
    python -m lucid.test.audit --axis grad,nonfinite
    python -m lucid.test.audit --subsystem nn.functional --select 'conv|pool'
    python -m lucid.test.audit --json audit.json --known lucid/test/audit/known.json
    python -m lucid.test.audit --list-uncovered      # what the harness cannot reach

Exit status is ``0`` when no defect survived, ``1`` when one did, ``2``
when the harness itself broke.  ``--list-uncovered`` is not decoration:
a census that cannot say what it missed is not a census, and that list is
the work queue for extending :mod:`~lucid.test.audit._specs`.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit import _axes, _probe, _surface
from lucid.test.audit._console import Console, Suppress, Timer, fmt_duration
from lucid.test.audit._result import STATUS_STYLE, Baseline, Finding, Report, Status

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_KNOWN = Path(__file__).with_name("known.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lucid.test.audit",
        description="Exhaustive correctness sweep over Lucid, excluding the model zoo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Statuses: PASS correct · FAIL a defect · TRNC the finite-difference probe "
            "was the limit, not the op · GAUG defined only up to a symmetry · "
            "VAC the check could not have failed · UNSP the op refused by design · "
            "SKIP the harness could not build inputs · KNWN accepted in the baseline."
        ),
    )
    sel = p.add_argument_group("selection")
    sel.add_argument(
        "--axis", default="all", help="comma-separated axis names, or 'all'"
    )
    sel.add_argument(
        "--subsystem", default="all", help="comma-separated subsystem keys, or 'all'"
    )
    sel.add_argument(
        "--select", default="", help="regular expression over symbol names"
    )
    sel.add_argument(
        "--limit", type=int, default=0, help="stop after N symbols (smoke run)"
    )

    run = p.add_argument_group("run")
    run.add_argument("--quick", action="store_true", help="fewer domains, looser sweep")
    run.add_argument(
        "--no-metal", action="store_true", help="skip every GPU comparison"
    )
    run.add_argument("--step", type=float, default=1e-5, help="finite-difference step")
    run.add_argument("--tolerance", type=float, default=2e-5, help="relative tolerance")
    run.add_argument(
        "--fail-fast", action="store_true", help="stop at the first defect"
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--json", type=Path, default=None, help="write the full report here"
    )
    out.add_argument("--known", type=Path, default=_DEFAULT_KNOWN, help="baseline file")
    out.add_argument(
        "--update-known",
        action="store_true",
        help="fold this run's defects into the baseline as accepted",
    )
    out.add_argument("--no-color", action="store_true")
    out.add_argument(
        "--quiet", action="store_true", help="only the summary and findings"
    )
    out.add_argument("--width", type=int, default=None)
    out.add_argument(
        "--show", default="fail,error,vacuous", help="statuses to list in full"
    )

    info = p.add_argument_group("information")
    info.add_argument("--list-axes", action="store_true")
    info.add_argument("--list-subsystems", action="store_true")
    info.add_argument(
        "--list-uncovered", action="store_true", help="symbols no axis reached"
    )
    return p


# ── information modes ────────────────────────────────────────────────────────


def _list_axes(console: Console) -> int:
    console.banner(
        "Audit axes", "each one has already caught a defect in this framework"
    )
    console.table(
        ["axis", "applies to", "question"],
        [[a.name, ", ".join(sorted(a.kinds)), a.summary] for a in _axes.ALL_AXES],
        always=True,
    )
    return 0


def _list_subsystems(console: Console) -> int:
    console.banner("Subsystems", "the audit denominator")
    symbols = _surface.enumerate_surface()
    rows = []
    for key in _surface.SUBSYSTEMS:
        of_key = [s for s in symbols if s.subsystem == key]
        if not of_key:
            continue
        inert = sum(1 for s in of_key if s.inert)
        rows.append(
            [
                key,
                str(len(of_key)),
                str(len(of_key) - inert),
                _surface.SUBSYSTEMS[key][1],
            ]
        )
    console.table(
        ["subsystem", "symbols", "stateful", "kind"],
        rows,
        ["l", "r", "r", "l"],
        always=True,
    )
    console.always("")
    for name, why in _surface.EXCLUDED.items():
        console.always("  " + console.paint(f"excluded  {name}", "grey") + f"  — {why}")
    return 0


# ── the run ──────────────────────────────────────────────────────────────────


def _selected_axes(spec: str) -> "list[_axes.Axis]":
    if spec.strip() in ("all", ""):
        return list(_axes.ALL_AXES)
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for name in wanted:
        axis = _axes.axis_by_name(name)
        if axis is None:
            raise SystemExit(
                f"unknown axis {name!r}; known axes: {', '.join(_axes.axis_names())}"
            )
        out.append(axis)
    return out


def _selected_symbols(args: argparse.Namespace) -> "list[_surface.Symbol]":
    subsystems = (
        None
        if args.subsystem.strip() in ("all", "")
        else [s.strip() for s in args.subsystem.split(",") if s.strip()]
    )
    symbols = _surface.enumerate_surface(subsystems)
    if args.select:
        pattern = re.compile(args.select)
        symbols = [s for s in symbols if pattern.search(s.qualname)]
    if args.limit:
        symbols = symbols[: args.limit]
    return symbols


def _tally(counts: "dict[Status, int]", console: Console) -> str:
    order = (
        Status.PASS,
        Status.FAIL,
        Status.VACUOUS,
        Status.TRUNCATION,
        Status.UNSUPPORTED,
        Status.SKIP,
        Status.KNOWN,
        Status.ERROR,
    )
    parts = []
    for status in order:
        n = counts.get(status, 0)
        if not n:
            continue
        label, colour = STATUS_STYLE[status]
        parts.append(console.paint(f"{label.strip().lower()} {n}", colour))
    return "  ".join(parts)


def run(args: argparse.Namespace, console: Console) -> Report:
    axes = _selected_axes(args.axis)
    symbols = _selected_symbols(args)
    metal = (not args.no_metal) and _probe.metal_available()
    ctx = _axes.Context(
        quick=args.quick, metal=metal, step=args.step, tolerance=args.tolerance
    )

    report = Report()
    report.surface_total = len(symbols)
    report.axes_run = [a.name for a in axes]
    report.config = {
        "axis": args.axis,
        "subsystem": args.subsystem,
        "select": args.select,
        "quick": args.quick,
        "metal": metal,
        "step": args.step,
        "tolerance": args.tolerance,
    }
    baseline = Baseline.load(args.known)

    console.banner(
        "Lucid — exhaustive audit",
        f"{len(axes)} axes · {len(symbols)} symbols · "
        f"metal {'on' if metal else 'off'} · model zoo excluded",
    )

    total_work = sum(sum(1 for s in symbols if a.applies(s)) for a in axes)
    overall = Timer()
    done_total = 0
    tick = 0
    stop = False

    for axis in axes:
        applicable = [s for s in symbols if axis.applies(s)]
        if not applicable:
            continue
        console.rule(f"{axis.name}  —  {axis.summary}", "cyan")
        counts: dict[Status, int] = dict.fromkeys(Status, 0)
        axis_timer = Timer()

        for index, symbol in enumerate(applicable, start=1):
            with Suppress(not console.quiet):
                try:
                    finding = axis.run(symbol, ctx)
                except KeyboardInterrupt:
                    raise
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - the harness must survive the survey
                    finding = Finding(
                        axis.name,
                        symbol.qualname,
                        Status.ERROR,
                        f"probe raised {type(exc).__name__}: {str(exc)[:90]}",
                    )
            finding = baseline.apply(finding)
            report.add(finding)
            counts[finding.status] += 1
            done_total += 1
            tick += 1

            if finding.status is Status.FAIL and not console.live:
                label, colour = STATUS_STYLE[Status.FAIL]
                console.always(
                    "  "
                    + console.paint(label, colour)
                    + f"  {symbol.qualname}  {finding.detail}"
                )

            if console.live and (tick % 3 == 0 or index == len(applicable)):
                console.live_block(
                    [
                        "  "
                        + console.paint(console.spinner(tick), "cyan")
                        + "  "
                        + console.paint(axis.name.ljust(12), "bold")
                        + console.paint(console.bar(index, len(applicable)), "cyan")
                        + f"  {index:>5}/{len(applicable)}"
                        + console.paint(
                            f"  {100 * index // max(len(applicable), 1):>3}%", "grey"
                        )
                        + console.paint(
                            f"   {axis_timer}  eta {axis_timer.eta(index, len(applicable))}",
                            "grey",
                        ),
                        "     "
                        + console.paint(symbol.qualname[: console.width - 8], "grey"),
                        "     " + _tally(counts, console),
                    ]
                )

            if args.fail_fast and finding.status.is_defect:
                stop = True
                break

        if console.live:
            console.live_done()
        console.write(
            "  "
            + console.paint(axis.name.ljust(12), "bold")
            + console.paint(f"{len(applicable):>5} symbols  ", "grey")
            + _tally(counts, console)
            + console.paint(f"   [{axis_timer}]", "grey")
        )
        if stop:
            console.write()
            console.always(
                console.paint("  stopped at the first defect (--fail-fast)", "yellow")
            )
            break

    report.duration = overall.elapsed
    _ = total_work
    return report


# ── reporting ────────────────────────────────────────────────────────────────


def summarise(report: Report, console: Console, show: "Sequence[str]") -> None:
    console.always("")
    console.rule("summary", "cyan")

    per_axis = report.counts_by_axis()
    rows = []
    for axis in report.axes_run:
        counts = per_axis.get(axis)
        if not counts:
            continue
        rows.append(
            [
                axis,
                str(sum(counts.values())),
                str(counts[Status.PASS]),
                str(counts[Status.FAIL]),
                str(counts[Status.VACUOUS]),
                str(counts[Status.TRUNCATION] + counts[Status.GAUGE]),
                str(counts[Status.UNSUPPORTED]),
                str(counts[Status.SKIP]),
            ]
        )
    console.table(
        ["axis", "run", "pass", "fail", "vacuous", "artefact", "unsupported", "skip"],
        rows,
        ["l", "r", "r", "r", "r", "r", "r", "r"],
        always=True,
    )

    reached, total, frac = report.coverage()
    filled, cells, cell_frac = report.cell_coverage()
    console.always("")
    console.always(
        "  "
        + console.paint("coverage".ljust(22), "grey")
        + console.paint(f"{reached}/{total} symbols reached ", "white")
        + console.paint(f"({100 * frac:.1f}%)", "cyan")
    )
    console.always(
        "  "
        + console.paint("symbol x axis".ljust(22), "grey")
        + console.paint(f"{filled}/{cells} cells filled ", "white")
        + console.paint(f"({100 * cell_frac:.1f}%)", "cyan")
        + console.paint("   — the measure that does not flatter", "grey")
    )
    console.always(
        "  "
        + console.paint("elapsed".ljust(22), "grey")
        + console.paint(fmt_duration(report.duration), "white")
    )

    wanted = {s.strip().lower() for s in show if s.strip()}
    for status in (Status.FAIL, Status.ERROR, Status.VACUOUS, Status.KNOWN):
        if status.value not in wanted:
            continue
        items = report.by_status(status)
        if not items:
            continue
        label, colour = STATUS_STYLE[status]
        console.always("")
        console.rule(f"{label.strip()} · {len(items)}", colour)
        for f in items:
            console.always(
                "  "
                + console.paint(label, colour)
                + "  "
                + console.paint(f.symbol.ljust(34), "white")
                + console.paint(f.detail, "grey")
            )

    console.always("")
    defects = report.defects
    if defects:
        console.always(
            console.paint(
                f"  {len(defects)} defect(s) — see the FAIL list above", "red", "bold"
            )
        )
    else:
        console.always(
            console.paint("  no defects survived verification", "green", "bold")
        )


def report_uncovered(console: Console, args: argparse.Namespace) -> int:
    """What the harness could not build inputs for — the work queue."""
    from lucid.test.audit import _specs

    symbols = _selected_symbols(args)
    ops = [s for s in symbols if s.kind in ("op", "method") and s.inert]
    without = [s for s in ops if not _specs.has_spec(s.short)]

    console.banner(
        "Symbols with no dedicated invocation spec",
        "they fall back to the generic ladder — fine for elementwise, not for the rest",
    )
    by_sub: dict[str, list[str]] = {}
    for s in without:
        by_sub.setdefault(s.subsystem, []).append(s.short)
    rows = [
        [
            key,
            str(len(names)),
            ", ".join(sorted(names)[:6]) + ("…" if len(names) > 6 else ""),
        ]
        for key, names in sorted(by_sub.items())
    ]
    console.table(
        ["subsystem", "count", "examples"], rows, ["l", "r", "l"], always=True
    )
    console.always("")
    console.always(
        console.paint(
            f"  {len(without)} of {len(ops)} callable symbols rely on the generic ladder.",
            "yellow",
        )
    )
    console.always(
        console.paint(
            "  That is not the same as uncovered: the ladder handles a plain unary or",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  binary op perfectly well, and most of the list above is exactly that.",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  The authoritative gap is the SKIP list of an actual run — start from",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  `--json report.json`, then add a family pattern to _specs.py to close",
            "grey",
        )
    )
    console.always(console.paint("  a whole group at once.", "grey"))
    return 0


# ── entry point ──────────────────────────────────────────────────────────────


def main(argv: "Sequence[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(
        colour=False if args.no_color else None, width=args.width, quiet=args.quiet
    )

    if args.list_axes:
        return _list_axes(console)
    if args.list_subsystems:
        return _list_subsystems(console)
    if args.list_uncovered:
        return report_uncovered(console, args)

    started = time.time()
    try:
        report = run(args, console)
    except KeyboardInterrupt:
        console.always("")
        console.always(console.paint("  interrupted — no report written", "yellow"))
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.always("")
        console.always(
            console.paint(f"  the audit harness failed: {exc!r}", "red", "bold")
        )
        return 2

    report.started = started
    summarise(report, console, args.show.split(","))

    if args.json is not None:
        report.write_json(args.json)
        console.always(console.paint(f"  report written to {args.json}", "grey"))
    if args.update_known:
        Baseline.load(args.known).save(args.known, report.defects)
        console.always(
            console.paint(
                f"  {len(report.defects)} defect(s) folded into {args.known}", "grey"
            )
        )
        return 0
    return 1 if report.defects else 0


if __name__ == "__main__":
    sys.exit(main())
