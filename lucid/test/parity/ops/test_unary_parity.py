"""Reference-parity tests for unary ops.

For each op we draw the same NumPy-seeded input, push it through both
Lucid and the reference framework, and assert element-wise agreement
within tight atol.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close

# (name, lucid_fn, ref_attr, sample_range, atol)
_UNARY_OPS: list[tuple[str, Callable, str, tuple[float, float], float]] = [
    ("abs", lambda t: t.abs(), "abs", (-2.0, 2.0), 1e-6),
    ("neg", lambda t: -t, "neg", (-2.0, 2.0), 1e-6),
    ("exp", lambda t: t.exp(), "exp", (-2.0, 2.0), 1e-5),
    ("log", lambda t: t.log(), "log", (0.1, 5.0), 1e-5),
    ("sqrt", lambda t: t.sqrt(), "sqrt", (0.1, 5.0), 1e-5),
    ("sin", lambda t: t.sin(), "sin", (-3.0, 3.0), 1e-5),
    ("cos", lambda t: t.cos(), "cos", (-3.0, 3.0), 1e-5),
    ("tanh", lambda t: t.tanh(), "tanh", (-2.0, 2.0), 1e-5),
    ("sigmoid", lambda t: t.sigmoid(), "sigmoid", (-3.0, 3.0), 1e-5),
    ("erf", lambda t: lucid.erf(t), "erf", (-2.0, 2.0), 1e-5),
    ("floor", lambda t: t.floor(), "floor", (-2.5, 2.5), 0.0),
    ("ceil", lambda t: t.ceil(), "ceil", (-2.5, 2.5), 0.0),
    ("round", lambda t: t.round(), "round", (-2.5, 2.5), 0.0),
    ("expm1", lambda t: lucid.expm1(t), "expm1", (-1.0, 1.0), 1e-5),
    ("log1p", lambda t: lucid.log1p(t), "log1p", (-0.5, 5.0), 1e-5),
    ("rsqrt", lambda t: lucid.rsqrt(t), "rsqrt", (0.1, 5.0), 1e-4),
]


@pytest.mark.parity
@pytest.mark.parametrize(
    "name,lucid_fn,ref_attr,rng,atol", _UNARY_OPS, ids=[op[0] for op in _UNARY_OPS]
)
def test_unary_parity(
    name: str,
    lucid_fn: Callable,
    ref_attr: str,
    rng: tuple[float, float],
    atol: float,
    ref: Any,
) -> None:
    np.random.seed(0)
    x = np.random.uniform(rng[0], rng[1], size=(4, 5)).astype(np.float32)
    lt = lucid.tensor(x.copy())
    rt = ref.tensor(x.copy())
    l_out = lucid_fn(lt)
    r_out = getattr(ref, ref_attr)(rt)
    assert_close(l_out, r_out, atol=atol)
