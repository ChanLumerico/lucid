"""CPU float32 erf and GELU: within one ulp of the true value, on a dense grid.

``erf`` on the CPU used libm's scalar ``erff``, which kept the exact GELU's
loop scalar.  It is now a branch-free double-precision polynomial
(``backend/cpu/ErfPoly.h``) that rounds to the correctly rounded float almost
everywhere (Linear CHA-9).  The GPT training parity rests on an exact erf, so
the bound is held here: at most one ulp from ``math.erf``.

The tanh-approximate GELU took a scalar ``tanhf`` per element on one core,
4 ms for 524k elements.  It now runs through vForce a tile at a time and is
held to the formula evaluated in float64.

GELU is held in absolute terms, scaled by the input: ``1 + erf`` and
``1 + tanh`` cancel for negative inputs, so a tiny output carries the
rounding of a value near one.  That is the formula's own precision, the
same before and after; a relative bound there would test the cancellation,
not the kernel.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

GRID = np.concatenate(
    [
        np.linspace(-6.0, 6.0, 400_001, dtype=np.float32),
        np.array([0.0, -0.0, 1.0, -1.0, 4.0, -4.0, 1e-30, -1e-30], np.float32),
    ]
)


def _assert_close_to_scale(
    got: np.ndarray, want: np.ndarray, x: np.ndarray, eps: float
) -> None:
    bound = eps * np.maximum(1.0, np.abs(x.astype(np.float64)))
    worst = float((np.abs(got.astype(np.float64) - want) / bound).max())
    assert worst <= 1.0, f"off by {worst:.2f}x the bound"


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(
        a.astype(np.float32).view(np.int32).astype(np.int64)
        - b.astype(np.float32).view(np.int32).astype(np.int64)
    )


def test_erf_is_within_one_ulp_of_the_correctly_rounded_value() -> None:
    got = lucid.erf(lucid.tensor(GRID)).numpy()
    want = np.array([math.erf(float(v)) for v in GRID]).astype(np.float32)
    assert int(_ulps(got, want).max()) <= 1


def test_erf_keeps_the_special_values() -> None:
    special = np.array([np.inf, -np.inf, np.nan, -0.0], np.float32)
    got = lucid.erf(lucid.tensor(special)).numpy()
    assert got[0] == 1.0 and got[1] == -1.0 and np.isnan(got[2])
    assert got[3] == 0.0 and np.signbit(got[3])


def test_exact_gelu_and_its_gradient_follow_erf() -> None:
    x64 = GRID.astype(np.float64)
    cdf = 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x64]))
    got = F.gelu(lucid.tensor(GRID)).numpy()
    _assert_close_to_scale(got, x64 * cdf, GRID, 2.0**-22)
    x = lucid.tensor(GRID, requires_grad=True)
    F.gelu(x).sum().backward()
    pdf = np.exp(-0.5 * x64 * x64) / math.sqrt(2.0 * math.pi)
    _assert_close_to_scale(x.grad.numpy(), cdf + x64 * pdf, GRID, 2.0**-21)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tanh_gelu_matches_its_formula(dtype: type) -> None:
    eps = 2.0**-22 if dtype is np.float32 else 2.0**-50
    x64 = GRID.astype(np.float64)
    c1, c2 = math.sqrt(2.0 / math.pi), 0.044715
    t = np.tanh(c1 * (x64 + c2 * x64**3))
    got = F.gelu(lucid.tensor(GRID.astype(dtype)), approximate="tanh").numpy()
    _assert_close_to_scale(got, 0.5 * x64 * (1.0 + t), GRID, eps)
    x = lucid.tensor(GRID.astype(dtype), requires_grad=True)
    F.gelu(x, approximate="tanh").sum().backward()
    dinner = c1 * (1.0 + 3.0 * c2 * x64 * x64)
    grad = 0.5 * (1.0 + t) + 0.5 * x64 * (1.0 - t * t) * dinner
    # 1 - t^2 cancels near t = +-1, multiplied by x * dinner (up to ~25):
    # an ulp of t becomes several of the gradient.
    _assert_close_to_scale(x.grad.numpy(), grad, GRID, 16 * eps)
