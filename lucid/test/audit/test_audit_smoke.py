"""The audit harness has to work, so it gets its own tests.

Deliberately not the full sweep — that is minutes long and its output is
a report rather than an assertion.  What is checked here is that the
machinery cannot rot silently: the surface enumerates, every axis runs
without raising, the statuses mean what the summary claims, and the two
discriminators that keep the FAIL list short still discriminate.

The last point matters most.  Without the truncation check a
finite-difference probe near a pole reports a defect in ``reciprocal``,
and without the vacuity check a softmax contracted against ``ones``
reports a pass having tested nothing.  Both have happened.
"""

import numpy as np
import pytest

import lucid
from lucid.test.audit import _axes, _probe, _specs, _surface
from lucid.test.audit._console import Console, _strip
from lucid.test.audit._result import Baseline, Finding, Report, Status

# ── surface ──────────────────────────────────────────────────────────────────


def test_surface_enumerates_and_excludes_the_model_zoo() -> None:
    symbols = _surface.enumerate_surface()
    assert len(symbols) > 500
    assert not any(s.qualname.startswith("lucid.models") for s in symbols)
    assert "lucid.models" in _surface.EXCLUDED


def test_stateful_symbols_are_marked_not_dropped() -> None:
    """They count towards the denominator; they are simply never called.

    A first sweep called ``set_grad_enabled`` with a tensor and poisoned
    every op after it alphabetically — 278 of them.
    """
    symbols = {s.qualname: s for s in _surface.enumerate_surface(["lucid"])}
    poisonous = symbols.get("lucid.set_grad_enabled")
    assert poisonous is not None, "must still be counted"
    assert not poisonous.inert, "must never be invoked"


def test_every_subsystem_key_resolves() -> None:
    symbols = _surface.enumerate_surface()
    seen = {s.subsystem for s in symbols}
    assert seen <= set(_surface.SUBSYSTEMS)
    assert "nn.functional" in seen and "lucid" in seen


# ── specs ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "conv2d",
        "max_pool2d",
        "adaptive_avg_pool2d",
        "batch_norm",
        "layer_norm",
        "embedding",
    ],
)
def test_the_families_that_used_to_be_skipped_have_specs(name: str) -> None:
    """Every convolution, pooling and normalisation was unreachable before."""
    assert _specs.has_spec(name), name
    assert next(_specs.invocations(name, "moderate"), None) is not None


def test_invocations_always_offer_a_fallback() -> None:
    """An unknown name must still get the generic ladder, not an empty iterator."""
    assert (
        next(_specs.invocations("a_name_that_does_not_exist", "moderate"), None)
        is not None
    )


def test_call_replaces_only_the_primary_argument() -> None:
    call = next(_specs.invocations("conv2d", "moderate"))
    swapped = call.with_primary(
        np.zeros(_probe.to_numpy(call.args[call.primary]).shape)
    )
    assert float(lucid.abs(swapped.args[swapped.primary]).max()) == 0.0
    assert len(swapped.args) == len(call.args)


# ── discriminators ───────────────────────────────────────────────────────────


def test_quadratic_shrink_separates_truncation_from_a_real_error() -> None:
    # Truncation: refining h by 10 cuts the disagreement by ~100.
    assert _probe.quadratic_shrink(2.6e-4, 2.6e-6)
    # A wrong formula does not move.
    assert not _probe.quadratic_shrink(1.0, 0.99)
    assert not _probe.quadratic_shrink(5.3e-4, 4.9e-4)


def test_covector_is_never_constant() -> None:
    """A constant seed makes softmax's gradient identically zero."""
    weights = _probe.covector(16)
    assert weights.std() > 0.1
    assert not np.allclose(weights, weights[0])


def test_vacuous_is_reported_for_a_gradient_that_cannot_fail() -> None:
    axis = _axes.GradientAxis()
    symbol = _surface.Symbol("lucid.zeros_like", "lucid", "op", lucid.zeros_like)
    finding = axis.run(symbol, _axes.Context(quick=True))
    assert finding.status in (Status.VACUOUS, Status.UNSUPPORTED, Status.SKIP)


# ── axes ─────────────────────────────────────────────────────────────────────

_AXIS_PROBES = [
    ("grad", "lucid.tanh", lucid.tanh),
    ("creategraph", "lucid.exp", lucid.exp),
    ("grad2", "lucid.sin", lucid.sin),
    ("nonfinite", "lucid.exp", lucid.exp),
    ("entry", "lucid.exp", lucid.exp),
    ("edge", "lucid.exp", lucid.exp),
]


@pytest.mark.parametrize("axis_name,qualname,obj", _AXIS_PROBES)
def test_axis_runs_and_returns_a_finding(
    axis_name: str, qualname: str, obj: object
) -> None:
    axis = _axes.axis_by_name(axis_name)
    assert axis is not None
    finding = axis.run(
        _surface.Symbol(qualname, "lucid", "op", obj), _axes.Context(quick=True)
    )
    assert isinstance(finding, Finding)
    assert finding.axis == axis_name
    assert finding.status is not Status.ERROR, finding.detail


def test_a_known_good_op_passes_the_gradient_axis() -> None:
    """Guard the instrument: if nothing can pass, nothing can fail either."""
    axis = _axes.GradientAxis()
    finding = axis.run(
        _surface.Symbol("lucid.tanh", "lucid", "op", lucid.tanh),
        _axes.Context(quick=True),
    )
    assert finding.status is Status.PASS, finding.detail


def test_axes_declare_which_kinds_they_apply_to() -> None:
    for axis in _axes.ALL_AXES:
        assert axis.name and axis.summary
        assert axis.kinds


# ── report ───────────────────────────────────────────────────────────────────


def test_report_coverage_counts_only_reached_symbols() -> None:
    report = Report(surface_total=4, axes_run=["grad"])
    report.add(Finding("grad", "a", Status.PASS))
    report.add(Finding("grad", "b", Status.SKIP))
    reached, total, frac = report.coverage()
    assert (reached, total) == (1, 4)
    assert frac == pytest.approx(0.25)


def test_cell_coverage_uses_the_symbol_times_axis_matrix() -> None:
    report = Report(surface_total=10, axes_run=["grad", "device"])
    report.add(Finding("grad", "a", Status.PASS))
    filled, cells, frac = report.cell_coverage()
    assert (filled, cells) == (1, 20)
    assert frac == pytest.approx(0.05)


def test_skip_does_not_count_as_coverage() -> None:
    assert not Status.SKIP.is_coverage
    assert not Status.UNSUPPORTED.is_coverage
    assert Status.PASS.is_coverage and Status.FAIL.is_coverage


def test_only_fail_is_a_defect() -> None:
    defects = [s for s in Status if s.is_defect]
    assert defects == [Status.FAIL]


def test_report_serialises() -> None:
    report = Report(surface_total=2, axes_run=["grad"])
    report.add(Finding("grad", "lucid.exp", Status.PASS, "rel 1e-11", {"rel": 1e-11}))
    payload = report.as_dict()
    assert payload["coverage"]["symbols_total"] == 2
    assert payload["findings"][0]["symbol"] == "lucid.exp"


# ── baseline ─────────────────────────────────────────────────────────────────


def test_baseline_downgrades_an_accepted_defect() -> None:
    baseline = Baseline({"nonfinite::lucid.sign": "convention, reviewed"})
    got = baseline.apply(Finding("nonfinite", "lucid.sign", Status.FAIL, "returned 0"))
    assert got.status is Status.KNOWN
    assert "convention" in got.detail


def test_baseline_leaves_an_unlisted_defect_alone() -> None:
    baseline = Baseline({})
    got = baseline.apply(Finding("grad", "lucid.exp", Status.FAIL, "rel 1.0"))
    assert got.status is Status.FAIL


def test_the_shipped_baseline_explains_every_entry() -> None:
    """An unexplained entry is indistinguishable from a silenced bug."""
    from pathlib import Path

    baseline = Baseline.load(Path(__file__).with_name("known.json"))
    for key, reason in baseline.entries.items():
        assert len(reason) > 40, f"{key} needs a real reason, got {reason!r}"


# ── console ──────────────────────────────────────────────────────────────────


def test_console_degrades_without_a_terminal() -> None:
    console = Console(colour=False, width=80)
    assert console.paint("x", "red") == "x"
    assert not console.live


def test_bar_and_strip_agree_on_width() -> None:
    console = Console(colour=False, width=80)
    assert len(_strip(console.bar(0, 10, 20))) == 20
    assert len(_strip(console.bar(10, 10, 20))) == 20
    assert len(_strip(console.bar(3, 10, 20))) == 20


def test_table_renders_without_a_terminal() -> None:
    console = Console(colour=False, width=80)
    console.table(["a", "bb"], [["1", "2"]])


# ── reach ────────────────────────────────────────────────────────────────────


def test_every_symbol_has_at_least_one_axis() -> None:
    """The property the tool is for.

    Reach went 57.3% -> 92.2% -> 100% as ``Tensor.*`` was resolved
    properly, the subsystems with no axis got one, and the smoke floor
    was widened to every callable.  If this regresses, some part of the
    framework has silently left the audit.
    """
    symbols = _surface.enumerate_surface()
    stranded = [
        s.qualname for s in symbols if not any(a.applies(s) for a in _axes.ALL_AXES)
    ]
    assert not stranded, f"{len(stranded)} symbols no axis reaches: {stranded[:10]}"


def test_tensor_methods_resolve_to_something_callable() -> None:
    """253 symbols had no axis until properties and functions were told apart."""
    methods = [s for s in _surface.enumerate_surface(["tensor"]) if s.inert]
    assert len(methods) > 200
    unresolved = [s.qualname for s in methods if _surface.resolve(s) is None]
    assert not unresolved, unresolved[:10]


def test_no_namespace_is_counted_as_a_symbol() -> None:
    """``nn.functional`` is a subsystem, not a member of ``nn``."""
    import types as _types

    symbols = _surface.enumerate_surface()
    namespaces = [s.qualname for s in symbols if isinstance(s.obj, _types.ModuleType)]
    assert not namespaces, namespaces


def test_subsystem_kinds_reach_their_dedicated_axis() -> None:
    by_name = {a.name: a for a in _axes.ALL_AXES}
    for subsystem, axis_name in (
        ("distributions", "distribution"),
        ("diffeq", "diffeq"),
        ("optim", "optim"),
        ("serialization", "serialize"),
    ):
        symbols = _surface.enumerate_surface([subsystem])
        assert symbols, subsystem
        assert any(by_name[axis_name].applies(s) for s in symbols), subsystem
