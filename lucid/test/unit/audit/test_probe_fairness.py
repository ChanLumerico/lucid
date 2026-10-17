"""Each audit cell asks the op its own question, not one the probe made up.

Between 2026-10-04 and 10-16 the checks part of CI failed on audit verdicts
that were about the probes rather than the framework:

* ``fmod`` / ``remainder`` / ``floor_divide`` were "missing" every integer
  dtype on the CPU.  The float companions truncated to a zero divisor, and
  the CPU refuses integer division by zero (LCD-159) where Metal answers.
* ``Tensor.clamp(self, *args)`` was handed ``x.clamp(x, y)``: every element
  on its lower bound, a corner, reported as a missing double backward.
* ``Tensor.unfold`` and ``F.unfold`` were compared as one op, which they are
  not, in the reference either.
* ``ChainedScheduler`` was handed a scheduler of another optimizer, which it
  now refuses as the reference does.
* ``HalfNormal``'s ``icdf`` round trip compared a clipped probability's
  quantile with the unclipped point.
* ``hessian`` / ``jacobian`` / ``vjp`` got a list where the annotation says
  ``tuple``, and ``register_kl`` got ``nn.Linear`` for a distribution class.
* ``sampling.py`` imported its private grid helpers under public names, so
  ``F.reflect_grid`` was audited as API and failed the gradient axis for
  reading only its input's shape.
"""

import functools

import pytest

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import (
    Context,
    DtypeAxis,
    GradientAxis,
    SecondGradientAxis,
)
from lucid.test.audit._axes_data import SchedulerAxis
from lucid.test.audit._axes_stability import RankAxis
from lucid.test.audit._axes_subsystem import DistributionAxis, SmokeAxis
from lucid.test.audit._result import Status


@functools.cache
def _symbols() -> dict[str, _surface.Symbol]:
    return {s.qualname: s for s in _surface.enumerate_surface()}


def _symbol(qualname: str) -> _surface.Symbol:
    return _symbols()[qualname]


def _context() -> Context:
    return Context(metal=_probe.metal_available())


@pytest.mark.parametrize(
    "qualname",
    [
        "Tensor.floor_divide",
        "Tensor.fmod",
        "Tensor.remainder",
        "lucid.fmod",
        "lucid.remainder",
    ],
)
def test_a_zero_divisor_refusal_is_not_missing_dtype_support(qualname: str) -> None:
    finding = DtypeAxis().run(_symbol(qualname), _context())
    assert finding.status == Status.PASS, finding.detail


def test_only_a_division_by_zero_counts_as_a_value_refusal() -> None:
    with pytest.raises(RuntimeError) as caught:
        lucid.fmod(
            lucid.tensor([3], dtype=lucid.int32), lucid.tensor([0], dtype=lucid.int32)
        )
    assert _probe.refuses_a_value(caught.value)
    assert _probe.refuses_a_value(ZeroDivisionError("integer division by zero"))
    assert not _probe.refuses_a_value(TypeError("dtype not supported: int8"))


def test_a_variadic_methods_operands_are_not_its_receivers_draw() -> None:
    symbol = _symbol("Tensor.clamp")
    fn = _surface.resolve(symbol)
    call = next(iter(_specs.invocations(symbol.short, "moderate", symbol.qualname, fn)))
    receiver, *rest = (_probe.to_numpy(a) for a in call.args)
    assert rest, "Tensor.clamp(self, *args) got no bound to compare"
    for bound in rest:
        assert bound is not None and not (bound == receiver).any()


@pytest.mark.parametrize("qualname", ["Tensor.clamp", "Tensor.clip"])
def test_clamp_is_not_probed_on_its_corner(qualname: str) -> None:
    for axis in (GradientAxis(), SecondGradientAxis()):
        finding = axis.run(_symbol(qualname), _context())
        assert finding.status == Status.PASS, f"{axis.name}: {finding.detail}"


def test_ops_that_only_share_a_name_are_not_compared() -> None:
    for qualname, labels in (
        ("Tensor.unfold", {"method"}),
        ("F.unfold", {"functional"}),
    ):
        got = {label for label, _ in _surface.counterparts(_symbol(qualname))}
        assert got == labels, (qualname, got)


def test_a_chained_scheduler_is_handed_schedulers_of_its_own_optimizer() -> None:
    finding = SchedulerAxis().run(_symbol("lucid.optim.ChainedScheduler"), _context())
    assert finding.status == Status.PASS, finding.detail


def test_the_icdf_round_trip_compares_only_points_it_did_not_clip() -> None:
    finding = DistributionAxis().run(
        _symbol("lucid.distributions.HalfNormal"), _context()
    )
    assert finding.status == Status.PASS, finding.detail


@pytest.mark.parametrize(
    "qualname",
    [
        "lucid.autograd.hessian",
        "lucid.autograd.jacobian",
        "lucid.autograd.vjp",
        "lucid.distributions.register_kl",
    ],
)
def test_signature_derived_calls_use_the_values_the_annotation_names(
    qualname: str,
) -> None:
    finding = SmokeAxis().run(_symbol(qualname), _context())
    assert finding.status == Status.PASS, finding.detail


@pytest.mark.parametrize(
    "helper",
    ["reflect_grid", "check_options", "composite_grid_sample", "affine_grid_3d"],
)
def test_private_grid_helpers_are_not_audited_as_api(helper: str) -> None:
    assert f"F.{helper}" not in _symbols()


def test_split_is_probed_with_sizes_its_axis_takes() -> None:
    # Since LCD-490 split refuses sizes that do not sum to the axis; the
    # derived ``(2, 2)`` fitted only a 1-D probe, and the rank axis lost it.
    finding = RankAxis().run(_symbol("lucid.split"), _context())
    assert finding.status == Status.PASS, finding.detail
