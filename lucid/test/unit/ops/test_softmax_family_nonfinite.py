"""The softmax family on rows whose max is not finite (LCD-286).

A max-shift reduction subtracts the row max before ``exp``.  When that max
is infinite, ``inf - inf`` is NaN and poisons the row, so each op has to
decide what an infinite max means — and the two devices decided differently:

* ``logsumexp`` answered NaN on both devices for a row holding +inf (the
  reference: +inf) and for an all -inf row (the reference: -inf).  The
  reference shifts an infinite max by 0; so does Lucid now.
* ``log_softmax`` on Metal went through MLX's guarded ``logsumexp`` and
  answered ``[-inf, nan, -inf]`` for ``[1, inf, 2]``, where the reference
  and the CPU shift by the raw max and answer all NaN — as ``softmax``
  does on every device.  Those Metal rows are strict xfails until the
  GpuBackend change lands.

Every row is checked against the reference in its own dtype, and on both
devices, so the two devices agree because each agrees with the reference.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available

_INF, _NAN = math.inf, math.nan

_ROWS = {
    "finite": [1.0, 2.0, 3.0],
    "pos_inf": [1.0, _INF, 2.0],
    "pos_and_neg_inf": [_INF, 1.0, -_INF],
    "two_pos_inf": [_INF, _INF, 1.0],
    "neg_inf_finite_max": [-_INF, 1.0, 2.0],
    "all_neg_inf": [-_INF, -_INF, -_INF],
    "nan": [1.0, _NAN, 2.0],
}

# Rows holding +inf, where the Metal log_softmax still answers MLX's guarded
# form until its GpuBackend change lands.  An all -inf row is NaN either way.
_POS_INF_MAX = {"pos_inf", "pos_and_neg_inf", "two_pos_inf"}

_OPS = {
    "softmax": (lambda x, d: F.softmax(x, d), lambda r, x, d: r.softmax(x, d)),
    "log_softmax": (
        lambda x, d: F.log_softmax(x, d),
        lambda r, x, d: r.log_softmax(x, d),
    ),
    "logsumexp": (
        lambda x, d: lucid.logsumexp(x, d),
        lambda r, x, d: r.logsumexp(x, d),
    ),
}

_DTYPES = {"float32": 1e-6, "float64": 1e-12, "float16": 2e-3}


def _cases():
    devices = ["cpu", "metal"] if metal_available() else ["cpu"]
    for device in devices:
        for dtype in _DTYPES:
            if device == "metal" and dtype == "float64":
                continue
            for op in _OPS:
                for row in _ROWS:
                    marks = ()
                    if device == "metal" and op == "log_softmax" and row in _POS_INF_MAX:
                        marks = (
                            pytest.mark.xfail(
                                strict=True,
                                reason="LCD-286 metal: GpuBackend.h edit pending user permission",
                            ),
                        )
                    yield pytest.param(
                        device, dtype, op, row, marks=marks, id=f"{device}-{dtype}-{op}-{row}"
                    )


def _check(got, want, dtype):
    tol = _DTYPES[dtype]
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want)), (got, want)
    assert np.array_equal(np.isinf(got) & (got > 0), np.isinf(want) & (want > 0)), (got, want)
    assert np.array_equal(np.isinf(got) & (got < 0), np.isinf(want) & (want < 0)), (got, want)
    finite = np.isfinite(want)
    np.testing.assert_allclose(got[finite], want[finite], rtol=tol, atol=tol)


@pytest.mark.parametrize(("device", "dtype", "op", "row"), list(_cases()))
@pytest.mark.parametrize("layout", ["last_axis", "leading_axis"])
def test_matches_reference(ref, device, dtype, op, row, layout):
    values = np.array(_ROWS[row], dtype=dtype)
    # ``leading_axis`` puts the row down a column, so the reduction runs
    # over a strided axis (the CPU's inner > 1 path) beside a finite column.
    dim = -1
    if layout == "leading_axis":
        values = np.stack([values, np.arange(3, dtype=dtype)], axis=1)
        dim = 0
    ours, theirs = _OPS[op]
    x = lucid.tensor(values, device=device)
    got = ours(x, dim).numpy().astype(np.float64)
    want = theirs(ref, ref.tensor(values), dim).numpy().astype(np.float64)
    _check(got, want, dtype)


@pytest.mark.parametrize("device", ["cpu", "metal"] if metal_available() else ["cpu"])
def test_logsumexp_gradient_on_infinite_max(ref, device):
    values = np.array([[1.0, _INF, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    x = lucid.tensor(values, device=device, requires_grad=True)
    lucid.logsumexp(x, 1).sum().backward()
    xr = ref.tensor(values, requires_grad=True)
    ref.logsumexp(xr, 1).sum().backward()
    _check(x.grad.numpy().astype(np.float64), xr.grad.numpy().astype(np.float64), "float32")
