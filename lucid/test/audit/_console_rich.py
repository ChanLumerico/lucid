"""A live per-subsystem display, when ``rich`` is installed.

The stdlib console in :mod:`~lucid.test.audit._console` shows one bar at
a time, for the axis currently running.  That is the wrong axis of the
work to watch: a sweep is 27 axes over 34 subsystems, and what a person
wants to know part-way through is *which package is being hammered and
how it is doing*, not which of 27 questions is currently being asked.

This module renders instead:

* one overall bar over every (symbol x axis) cell, with elapsed and ETA;
* one bar per subsystem, each with its own running pass / fail / skip;
* a footer naming the axis and symbol in flight.

``rich`` is optional.  :func:`build` returns ``None`` when it is not
installed, when the output is redirected, or under ``--quiet`` /
``--no-color``, and the caller keeps the stdlib display — the audit has
never needed a third-party package to run and still does not.

Install it with ``pip install lucid-dl[audit]``.
"""

import sys
from typing import TYPE_CHECKING, Any

from lucid.test.audit._result import Status

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from lucid.test.audit._axes import Axis
    from lucid.test.audit._console import Console
    from lucid.test.audit._result import Finding
    from lucid.test.audit._surface import Symbol

try:  # pragma: no cover - exercised by whether the extra is installed
    from rich.console import Console as RichConsole
    from rich.console import Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.text import Text

    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


def available() -> bool:
    """bool: Whether ``rich`` can be imported."""
    return _AVAILABLE


#: Cells between redraws.  A full re-render of every bar costs more than
#: some of the cells being measured, and nothing is lost by batching:
#: the counters are exact, only the frame rate is coarse.
_REFRESH_EVERY = 5


class RichReporter:
    """Live progress, one bar per subsystem.

    Parameters
    ----------
    axes : sequence of Axis
        The axes this run will execute, used only for the total.
    symbols : sequence of Symbol
        Every symbol in scope.

    Notes
    -----
    Totals are computed the same way the runner computes its own work
    total — ``axis.applies(symbol)`` summed over both — so the bars
    cannot drift from what actually runs.
    """

    def __init__(self, axes: "Sequence[Axis]", symbols: "Sequence[Symbol]") -> None:
        self._counts: "dict[str, dict[str, int]]" = {}
        totals: "dict[str, int]" = {}
        for symbol in symbols:
            applicable = sum(1 for axis in axes if axis.applies(symbol))
            if applicable:
                totals[symbol.subsystem] = totals.get(symbol.subsystem, 0) + applicable
        self._totals = dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

        self._overall = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=34, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta"),
            TimeRemainingColumn(),
            expand=False,
        )
        self._per_subsystem = Progress(
            TextColumn("  [dim]{task.description}"),
            BarColumn(bar_width=22, complete_style="blue", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("{task.fields[tally]}"),
            expand=False,
        )
        self._footer = Text("")
        self._overall_task = self._overall.add_task(
            "overall", total=sum(self._totals.values())
        )
        self._tasks: "dict[str, Any]" = {}
        for name, total in self._totals.items():
            self._counts[name] = {"pass": 0, "fail": 0, "other": 0}
            self._tasks[name] = self._per_subsystem.add_task(
                name.ljust(max(len(k) for k in self._totals)),
                total=total,
                tally=Text(""),
            )
        # Pinned to the real stdout and refreshed by hand.  The runner
        # wraps every probe in ``Suppress``, which swaps ``sys.stdout``
        # for /dev/null so an op's warnings do not bury the display — and
        # the sweep spends nearly all of its time inside that window, so
        # a background refresh thread writes almost every frame into the
        # void.  The first version of this drew exactly one frame, at the
        # start, and then sat still for the whole run.
        self._live = Live(
            self._render(),
            console=RichConsole(file=sys.stdout),
            transient=False,
            auto_refresh=False,
        )
        self._since_refresh = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def __enter__(self) -> "RichReporter":
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        self._live.update(self._render(), refresh=True)
        self._live.__exit__(exc_type, exc, tb)

    # ── updates ──────────────────────────────────────────────────────────────

    def record(self, symbol: "Symbol", axis_name: str, finding: "Finding") -> None:
        """Advance the bars for one completed cell."""
        bucket = self._counts.get(symbol.subsystem)
        if bucket is None:  # a subsystem with no applicable axis
            return
        if finding.status is Status.PASS:
            bucket["pass"] += 1
        elif finding.status.is_defect:
            bucket["fail"] += 1
        else:
            bucket["other"] += 1

        task = self._tasks[symbol.subsystem]
        self._per_subsystem.update(task, advance=1, tally=self._tally(bucket))
        self._overall.advance(self._overall_task, 1)
        self._footer = Text.assemble(
            ("  ", ""),
            (axis_name, "bold cyan"),
            ("  ", ""),
            (symbol.qualname, "dim"),
        )
        # Throttled: a full re-render of 34 bars per cell would cost more
        # than some of the cells do.
        self._since_refresh += 1
        if self._since_refresh >= _REFRESH_EVERY:
            self._since_refresh = 0
            self._live.update(self._render(), refresh=True)

    def defect(self, label: str, qualname: str, detail: str) -> None:
        """Print a defect above the live region, as it happens.

        Assembled as :class:`~rich.text.Text` rather than a markup
        string.  Findings quote what they saw, and what they saw is full
        of square brackets — ``cpu only ['bool'], metal only []`` — which
        rich would read as markup tags and either mangle or raise on.
        """
        line = Text()
        line.append(f"  {label}  ", style="bold red")
        line.append(qualname.ljust(34), style="bright_white")
        line.append(detail, style="dim")
        self._live.console.print(line)
        self._live.refresh()

    # ── rendering ────────────────────────────────────────────────────────────

    @staticmethod
    def _tally(bucket: "dict[str, int]") -> "Text":
        out = Text()
        out.append(f"  {bucket['pass']:>4} ok", style="green")
        if bucket["fail"]:
            out.append(f"  {bucket['fail']:>3} fail", style="bold red")
        if bucket["other"]:
            out.append(f"  {bucket['other']:>4} –", style="dim")
        return out

    def _render(self) -> "Group":
        return Group(self._overall, Text(""), self._per_subsystem, self._footer)


def build(
    console: "Console", axes: "Sequence[Axis]", symbols: "Sequence[Symbol]"
) -> "RichReporter | None":
    """A live reporter, or ``None`` to keep the stdlib display.

    ``console.live`` already encodes "a terminal that can be animated and
    a user who did not ask for quiet", so this defers to it rather than
    re-deriving the same conditions and disagreeing with it.
    """
    if not _AVAILABLE or not console.live:
        return None
    return RichReporter(axes, symbols)


__all__ = ["RichReporter", "available", "build"]
