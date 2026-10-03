"""Every op the compile matrix replays, replayed through a Core ML package.

``test_op_parity`` checks one op at a time on the input it traced with,
which cannot see an answer frozen into the package.  This drives the cases
of :mod:`lucid.test.unit.compile._op_matrix` instead — 200 of them, float32,
built on the CPU as an export is — and asks each package about a *second*
input, against eager on that input.

Its first run found, among 113 passing cases: ``2.0 ** x`` exported as
``x ** 2`` (Core ML's optimizer turns a ``pow`` with a constant 2 into a
square, whichever side the 2 is on); ``one_hot`` of float indices refused
at parse time; the CPU ``tensordot`` never recorded by the tracer, so its
result was a constant of the example; and sixteen ops with no translation.

A refusal is allowed only where :data:`EXPECTED_REFUSAL` says why, and an
entry whose case starts exporting fails as stale — delete it.
"""

import numpy as np
import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.autograd._grad_mode import no_grad
from lucid.compile import _tracing
from lucid.coreml._build import UnsupportedOp
from lucid.test.unit.compile import _op_matrix as M

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)

# Cases not run at all.  ``x @ (x @ x)`` never finishes loading — Core ML's
# CPU graph compiler spins (CHA-20) — and a model that writes it still hangs.
# ``matrix_power`` used to; it multiplies on the left now and is run below.
NEVER_RUN: dict[str, str] = {}

# Cases whose export is refused, and why.
EXPECTED_REFUSAL: dict[str, str] = {
    "astype_f32": "a cast to the input's own dtype is the input: nothing reads it",
    "nextafter": "no MIL op, and no exact composition of MIL ops",
    "cumprod": "MIL has cumsum only; exp of a log-cumsum is not exact",
    "cummax": "no running maximum in MIL",
    "cummin": "no running minimum in MIL",
    "det": "no determinant in MIL",
    "inv": "no matrix inverse in MIL",
    "batch_norm_train": "batch statistics are training; a package infers",
    "batch_norm3d_train": "batch statistics are training; a package infers",
    "interp_nearest3d": "MIL's nearest-neighbour upsampling is 2-D",
}


class _Apply(nn.Module):
    """One case as a module; a boolean result comes out as float32, which
    is what a package can return."""

    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> object:
        leaves = [
            t.to(lucid.float32) if t.dtype == lucid.bool_ else t
            for t in M._leaves(self.fn(x))  # type: ignore[operator]
        ]
        return leaves[0] if len(leaves) == 1 else tuple(leaves)


def _names() -> list[str]:
    return [
        c.name
        for c in M.CASES
        if "f32" in c.dtypes
        and not c.random
        and M.refusal(c.name, "f32", "cpu") is None
        and c.name not in NEVER_RUN
    ]


def _inputs(case: M.Case) -> tuple[lucid.Tensor, lucid.Tensor]:
    return tuple(  # type: ignore[return-value]
        M.make_input(case.kind, "f32", case.shape, seed).to("cpu") for seed in (1, 2)
    )


@pytest.mark.parametrize("name", _names())
def test_a_package_answers_a_new_input_as_eager_does(
    name: str, tmp_path: object
) -> None:
    case = M.CASE_BY_NAME[name]
    with M.on_device("cpu"):
        traced_on, asked = _inputs(case)
        model = _Apply(case.fn).eval()
        want = M._leaves(model(asked))  # builds the recipe's constants first
        path = f"{tmp_path}/op.mlpackage"
        if name in EXPECTED_REFUSAL:
            try:
                cml.export(model, traced_on, path).close()
            except UnsupportedOp, ValueError:
                return
            pytest.fail("exports now — delete its EXPECTED_REFUSAL entry")
        package = cml.export(model, traced_on, path)
        try:
            answer = package.predict(asked)
        finally:
            package.close()
    got = list(answer.values()) if isinstance(answer, dict) else [answer]
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        gn = g.numpy().astype(np.float64)
        wn = w.numpy().astype(np.float64)
        assert gn.size == wn.size, f"out[{i}] {gn.shape} vs {wn.shape}"
        gn = gn.reshape(wn.shape)
        np.testing.assert_array_equal(np.isnan(gn), np.isnan(wn), f"out[{i}] NaNs")
        finite = np.isfinite(wn)
        if not finite.any():
            continue
        scale = float(np.max(np.abs(wn[finite]))) or 1.0
        diff = float(np.max(np.abs(gn[finite] - wn[finite]))) / scale
        assert diff < 1e-4, f"out[{i}] relative difference {diff:.3g}"


def test_the_tables_name_real_cases() -> None:
    for name in [*NEVER_RUN, *EXPECTED_REFUSAL]:
        assert name in M.CASE_BY_NAME, name


@pytest.mark.parametrize("dtype", ["f32", "i64", "i32", "bool"])
def test_every_cpu_result_is_recorded_by_the_tracer(dtype: str) -> None:
    """An op the CPU tracer does not see becomes a constant of the example.

    ``tensordot`` recorded itself on the GPU path only, and an export —
    which traces on the CPU — froze its result into the package.  A cast
    to the input's own dtype returns the input, and is not an op.
    """
    unseen = []
    with M.on_device("cpu"):
        for case in M.CASES:
            if dtype not in case.dtypes or M.refusal(case.name, dtype, "cpu"):
                continue
            x = M.make_input(case.kind, dtype, case.shape, 1).to("cpu")
            M._leaves(case.fn(x))  # the recipe's constants, outside the trace
            with no_grad():
                with _tracing() as tracer:
                    out = case.fn(x)
            for t in M._leaves(out):
                if t._impl is not x._impl and tracer.lookup_id(t._impl) is None:
                    unseen.append(case.name)
    assert not unseen, f"results the CPU tracer never recorded: {unseen}"
