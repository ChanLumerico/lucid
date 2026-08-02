"""``python -m lucid.test.audit`` — the exhaustive sweep.

Installed as ``lucid-audit``; ``python -m lucid.test.audit`` is the same
program and works without reinstalling.

Examples
--------
::

    lucid-audit                          # everything
    lucid-audit --coverage               # reach and depth, running nothing
    lucid-audit --quick                  # fewer domains, smaller probes
    lucid-audit --axis grad,nonfinite
    lucid-audit --subsystem nn.functional --select 'conv|pool'
    lucid-audit --json audit.json --known lucid/test/audit/known.json
    lucid-audit --list-uncovered         # what the harness cannot reach

Exit status is ``0`` when no defect survived, ``1`` when one did, ``2``
when the harness itself broke.  ``--list-uncovered`` is not decoration:
a census that cannot say what it missed is not a census, and that list is
the work queue for extending :mod:`~lucid.test.audit._specs`.
"""

import argparse
import contextlib
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit import _axes, _console_rich, _probe, _surface
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
    info.add_argument(
        "--coverage",
        action="store_true",
        help="measure reach and depth against an independent walk (runs nothing)",
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


def _report_coverage(console: Console) -> int:
    """What fraction of Lucid this tool can speak about, measured not claimed.

    Two numbers, because they fail independently and only one of them is
    usually quoted:

    *Reach* is the surface against an **independent** recursive walk of
    the package.  The enumeration used to name its modules in a literal
    and the literal named nineteen; the walk finds a hundred and thirty,
    and the difference was 26% of the framework — including all of
    ``utils.transforms`` and ``nn.init`` — sitting silently outside the
    denominator while the tool reported a coverage figure over the rest.
    Deriving the check from a different traversal than the thing it
    checks is the whole point; a self-consistent count would have agreed
    with itself at 73.8% forever.

    *Depth* is how many reached symbols get an axis that can actually
    fail, rather than only the smoke axis.  Reaching a symbol and calling
    it once is not verifying it, and the two must not be added together.
    """
    console.banner("Coverage", "measured against an independent walk of the package")

    symbols = _surface.enumerate_surface()
    walked = _surface.independent_walk()
    reached = {id(s.obj) for s in symbols}
    absent = {name for oid, name in walked.items() if oid not in reached}
    reach = 100.0 * (len(walked) - len(absent)) / max(len(walked), 1)

    per_symbol = {
        s.qualname: [a.name for a in _axes.ALL_AXES if a.applies(s)] for s in symbols
    }
    cells = sum(len(v) for v in per_symbol.values())
    shallow = [s for s in symbols if per_symbol[s.qualname] == ["smoke"]]
    stateful_shallow = sum(1 for s in shallow if not s.inert)
    depth = 100.0 * (len(symbols) - len(shallow)) / max(len(symbols), 1)

    console.table(
        ["measure", "value", "of", "%"],
        [
            [
                "reach — objects the surface enumerates",
                str(len(walked) - len(absent)),
                str(len(walked)),
                f"{reach:.1f}",
            ],
            [
                "depth — symbols with an axis past smoke",
                str(len(symbols) - len(shallow)),
                str(len(symbols)),
                f"{depth:.1f}",
            ],
            ["cells — symbol x applicable axis", str(cells), "", ""],
        ],
        ["l", "r", "r", "r"],
        always=True,
    )
    console.always("")
    if absent:
        console.always(
            console.paint(
                f"  {len(absent)} object(s) the walk found and the surface did not:",
                "red",
            )
        )
        for name in sorted(absent)[:20]:
            console.always(f"      {name}")
    else:
        console.always(
            console.paint(
                "  every public object the walk found is in the surface.", "green"
            )
        )
    console.always(
        f"  {len(shallow)} symbol(s) reach only the smoke axis, of which "
        f"{stateful_shallow} mutate process state and are called under a guard "
        "because no numeric axis can express them."
    )
    console.always("")

    rows = []
    for key in _surface.SUBSYSTEMS:
        of_key = [s for s in symbols if s.subsystem == key]
        if not of_key:
            continue
        thin = sum(1 for s in of_key if per_symbol[s.qualname] == ["smoke"])
        rows.append(
            [
                key,
                str(len(of_key)),
                str(len(of_key) - thin),
                f"{100.0 * (len(of_key) - thin) / len(of_key):.0f}",
                str(sum(len(per_symbol[s.qualname]) for s in of_key)),
            ]
        )
    console.table(
        ["subsystem", "symbols", "with an axis", "%", "cells"],
        rows,
        ["l", "r", "r", "r", "r"],
        always=True,
    )
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
    report.applicable_cells = total_work
    overall = Timer()
    done_total = 0
    tick = 0
    stop = False

    # One bar per subsystem when ``rich`` is installed, one bar per axis
    # otherwise.  ``rich`` is an optional extra, so both paths stay live.
    reporter = _console_rich.build(console, axes, symbols)
    with contextlib.ExitStack() as stack:
        if reporter is not None:
            stack.enter_context(reporter)

        for axis in axes:
            applicable = [s for s in symbols if axis.applies(s)]
            if not applicable:
                continue
            if reporter is None:
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

                if reporter is not None:
                    reporter.record(symbol, axis.name, finding)
                    if finding.status is Status.FAIL:
                        label, _ = STATUS_STYLE[Status.FAIL]
                        reporter.defect(label, symbol.qualname, finding.detail)
                elif finding.status is Status.FAIL and not console.live:
                    label, colour = STATUS_STYLE[Status.FAIL]
                    console.always(
                        "  "
                        + console.paint(label, colour)
                        + f"  {symbol.qualname}  {finding.detail}"
                    )

                if (
                    reporter is None
                    and console.live
                    and (tick % 3 == 0 or index == len(applicable))
                ):
                    console.live_block(
                        [
                            "  "
                            + console.paint(console.spinner(tick), "cyan")
                            + "  "
                            + console.paint(axis.name.ljust(12), "bold")
                            + console.paint(console.bar(index, len(applicable)), "cyan")
                            + f"  {index:>5}/{len(applicable)}"
                            + console.paint(
                                f"  {100 * index // max(len(applicable), 1):>3}%",
                                "grey",
                            )
                            + console.paint(
                                f"   {axis_timer}  "
                                f"eta {axis_timer.eta(index, len(applicable))}",
                                "grey",
                            ),
                            "     "
                            + console.paint(
                                symbol.qualname[: console.width - 8], "grey"
                            ),
                            "     " + _tally(counts, console),
                        ]
                    )

                if args.fail_fast and finding.status.is_defect:
                    stop = True
                    break

            if reporter is None:
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
                    console.paint(
                        "  stopped at the first defect (--fail-fast)", "yellow"
                    )
                )
                break

    report.duration = overall.elapsed
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
        + console.paint("applicable cells".ljust(22), "grey")
        + console.paint(f"{filled}/{cells} produced a verdict ", "white")
        + console.paint(f"({100 * cell_frac:.1f}%)", "cyan")
        + console.paint("   — skips and refusals do not count", "grey")
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
    from lucid.test.audit import _autospec, _specs

    symbols = _selected_symbols(args)
    ops = [s for s in symbols if s.kind in ("op", "method") and s.inert]
    without = [s for s in ops if not _specs.has_spec(s.short, _surface.resolve(s))]

    console.banner(
        "Symbols with no invocation, hand-written or derived",
        "not even their own signature says enough to call them",
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
            f"  {len(without)} of {len(ops)} callable symbols fall through every tier.",
            "yellow" if without else "green",
        )
    )
    for symbol in without[:12]:
        console.always(
            "      "
            + console.paint(symbol.qualname, "grey")
            + f"  — {_autospec.explain(_surface.resolve(symbol), symbol.short)}"
        )
    console.always("")
    console.always(
        console.paint(
            "  Each reason names the parameter that stopped it.  A required argument",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  the derivation has no value for is closed by one entry in _autospec's",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  name table, which then covers every other op that spells it the same",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  way — writing a per-op spec is the last resort, not the first.",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  Reaching a symbol is still not verifying it: the authoritative gap is",
            "grey",
        )
    )
    console.always(
        console.paint(
            "  the SKIP list of an actual run, from `--json report.json`.",
            "grey",
        )
    )
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
    if args.coverage:
        return _report_coverage(console)
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
