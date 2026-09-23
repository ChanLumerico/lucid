"""The audit compares against a recorded set, not just its own counts.

An absolute count cannot show a regression.  An op that stops being
reachable moves one cell from ``pass`` to ``unsupported`` among fifteen
hundred already-unsupported ones, and nothing in the summary changes
visibly — so "did my change break something" had no answer that did not
depend on remembering yesterday's numbers.

``coverage.json`` records which ``(axis, symbol)`` cells produced a
verdict.  Every run is diffed against it, and a cell that used to answer
and no longer does is reported as a regression.  That is the direction
that matters: the op is still there, the audit simply cannot reach it any
more, which is exactly what a refactor breaks without breaking a test.

Only verdicts are recorded.  ``skip`` and ``unsupported`` carry probe
details that move for reasons unrelated to the framework, and a baseline
that churns is a baseline nobody re-reads.
"""

import json

import pytest

from lucid.test.audit.__main__ import (
    _answered,
    load_coverage,
    report_coverage_diff,
    save_coverage,
)
from lucid.test.audit._console import Console
from lucid.test.audit._result import Finding, Report, Status


def _report(*cells: "tuple[str, str, Status]") -> Report:
    report = Report()
    for axis, symbol, status in cells:
        report.add(Finding(axis, symbol, status, "detail"))
    return report


def _quiet() -> Console:
    return Console(colour=False, quiet=False)


# ── what gets recorded ────────────────────────────────────────────────────────


def test_only_verdicts_are_recorded() -> None:
    report = _report(
        ("grad", "lucid.exp", Status.PASS),
        ("grad", "lucid.log", Status.FAIL),
        ("grad", "lucid.sin", Status.SKIP),
        ("grad", "lucid.cos", Status.UNSUPPORTED),
        ("grad", "lucid.tan", Status.NOT_APPLICABLE),
    )
    assert _answered(report) == {
        "grad::lucid.exp": "pass",
        "grad::lucid.log": "fail",
    }


def test_a_baseline_round_trips(tmp_path) -> None:
    path = tmp_path / "coverage.json"
    report = _report(("grad", "lucid.exp", Status.PASS))
    assert save_coverage(path, report) == 1
    assert load_coverage(path) == {"grad::lucid.exp": "pass"}


def test_a_missing_baseline_reads_as_nothing_to_compare(tmp_path) -> None:
    assert load_coverage(tmp_path / "absent.json") is None


def test_a_corrupt_baseline_does_not_take_the_run_down(tmp_path) -> None:
    path = tmp_path / "coverage.json"
    path.write_text("{ this is not json")
    assert load_coverage(path) is None


# ── what the diff reports ─────────────────────────────────────────────────────


def test_an_unchanged_run_reports_no_drift(capsys) -> None:
    report = _report(("grad", "lucid.exp", Status.PASS))
    regressions = report_coverage_diff(report, _answered(report), _quiet())
    assert regressions == 0
    assert "unchanged" in capsys.readouterr().out


def test_a_cell_that_stops_answering_is_a_regression(capsys) -> None:
    """The case the whole file exists for: the op is still exported, the
    audit just cannot reach it any more."""
    recorded = {"grad::lucid.exp": "pass", "grad::lucid.log": "pass"}
    now = _report(
        ("grad", "lucid.exp", Status.PASS),
        ("grad", "lucid.log", Status.UNSUPPORTED),
    )
    regressions = report_coverage_diff(now, recorded, _quiet())
    assert regressions == 1
    out = capsys.readouterr().out
    assert "LOST" in out and "lucid.log" in out


def test_a_pass_turning_into_a_failure_is_a_regression(capsys) -> None:
    recorded = {"grad::lucid.exp": "pass"}
    now = _report(("grad", "lucid.exp", Status.FAIL))
    assert report_coverage_diff(now, recorded, _quiet()) == 1
    assert "WORSE" in capsys.readouterr().out


def test_a_newly_answered_cell_is_progress_not_a_regression(capsys) -> None:
    recorded = {"grad::lucid.exp": "pass"}
    now = _report(
        ("grad", "lucid.exp", Status.PASS),
        ("grad", "lucid.log", Status.PASS),
    )
    assert report_coverage_diff(now, recorded, _quiet()) == 0
    out = capsys.readouterr().out
    assert "NEW" in out and "update-coverage" in out


def test_a_failure_that_was_already_a_failure_is_not_new(capsys) -> None:
    """The defect list already reports it; the coverage diff is about
    reachability moving, not about a known failure persisting."""
    recorded = {"grad::lucid.exp": "fail"}
    now = _report(("grad", "lucid.exp", Status.FAIL))
    assert report_coverage_diff(now, recorded, _quiet()) == 0


def test_regressions_and_progress_are_reported_together(capsys) -> None:
    recorded = {"grad::a": "pass", "grad::b": "pass"}
    now = _report(("grad", "a", Status.PASS), ("grad", "c", Status.PASS))
    assert report_coverage_diff(now, recorded, _quiet()) == 1
    out = capsys.readouterr().out
    assert "LOST" in out and "NEW" in out


# ── the checked-in baseline ───────────────────────────────────────────────────


def test_the_repository_baseline_is_readable_and_substantial() -> None:
    import pathlib

    path = pathlib.Path(__file__).parents[2] / "audit" / "coverage.json"
    if not path.exists():
        pytest.skip("no baseline recorded in this checkout")
    recorded = load_coverage(path)
    assert recorded is not None
    assert len(recorded) > 5000, len(recorded)
    assert all("::" in key for key in recorded)
    payload = json.loads(path.read_text())
    assert payload["cells"] == len(recorded)
