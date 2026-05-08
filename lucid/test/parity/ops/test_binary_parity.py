"""Reference parity for binary ops."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close


# (name, lucid_fn, ref_attr, low, high, atol)
_BIN_OPS: list[tuple[str, Callable, str, float, float, float]] = [
    ("add",       lambda a, b: a + b,             "add",       -2.0, 2.0, 1e-6),
    ("sub",       lambda a, b: a - b,             "sub",       -2.0, 2.0, 1e-6),
    ("mul",       lambda a, b: a * b,             "mul",       -2.0, 2.0, 1e-6),
    ("div",       lambda a, b: a / b,             "div",        0.5, 2.0, 1e-5),
    ("pow",       lambda a, b: a ** b,            "pow",        0.5, 2.0, 1e-4),
    ("maximum",   lambda a, b: lucid.maximum(a, b),"maximum",   -2.0, 2.0, 0.0),
    ("minimum",   lambda a, b: lucid.minimum(a, b),"minimum",   -2.0, 2.0, 0.0),
    ("atan2",     lambda a, b: lucid.atan2(a, b),  "atan2",     -2.0, 2.0, 1e-5),
    ("hypot",     lambda a, b: lucid.hypot(a, b),  "hypot",      0.1, 2.0, 1e-4),
    ("logaddexp", lambda a, b: lucid.logaddexp(a, b),"logaddexp",-2.0, 2.0, 1e-5),
]


@pytest.mark.parity
@pytest.mark.parametrize("name,lucid_fn,ref_attr,low,high,atol", _BIN_OPS,
                         ids=[op[0] for op in _BIN_OPS])
def test_binary_parity(
    name: str,
    lucid_fn: Callable,
    ref_attr: str,
    low: float,
    high: float,
    atol: float,
    ref: Any,
) -> None:
    np.random.seed(0)
    shape = (4, 5)
    a = np.random.uniform(low, high, size=shape).astype(np.float32)
    b = np.random.uniform(low, high, size=shape).astype(np.float32)
    la = lucid.tensor(a.copy()); lb = lucid.tensor(b.copy())
    ra = ref.tensor(a.copy()); rb = ref.tensor(b.copy())
    l_out = lucid_fn(la, lb)
    r_out = getattr(ref, ref_attr)(ra, rb)
    assert_close(l_out, r_out, atol=atol)


@pytest.mark.parity
def test_matmul_parity(ref: Any) -> None:
    np.random.seed(0)
    a = np.random.uniform(-1.0, 1.0, size=(8, 16)).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, size=(16, 4)).astype(np.float32)
    l_out = lucid.tensor(a) @ lucid.tensor(b)
    r_out = ref.tensor(a) @ ref.tensor(b)
    assert_close(l_out, r_out, atol=1e-4)
