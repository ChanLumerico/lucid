"""A NumPy scalar operand is the Python number it holds.

``lucid_tensor == np.float32(1)`` returned a NumPy ndarray, and
``lucid_tensor + np.float32(1)`` raised ``TypeError``: a NumPy scalar other
than ``float64`` (which derives from ``float``) registers with
:mod:`numbers` without deriving from a builtin, so the operand checks saw
neither a tensor nor a number, and ``==`` returned ``NotImplemented`` for
NumPy to answer.  :func:`lucid._promotion.as_number` now turns it into the
Python number at the operand boundary, a weak scalar as the reference takes
it.  The oracle needs no reference framework: every spelling of every
binary op must give exactly what it gives for ``scalar.item()``.
"""

import operator
from collections.abc import Callable

import numpy as np
import pytest

import lucid

_REAL = [np.float16(1.5), np.float32(2.5), np.int8(3), np.int64(-2), np.uint8(4)]
_INTEGER = [np.int8(3), np.int64(2), np.uint8(1)]

_ARITHMETIC = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
]
_COMPARISON = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]
_BITWISE = [operator.and_, operator.or_, operator.xor, operator.lshift]

# The other spellings of one op: the method, the free function, in place.
_SPELLINGS: dict[str, Callable[[lucid.Tensor, object], lucid.Tensor]] = {
    "method": lambda t, s: t.mul(s),
    "function": lambda t, s: lucid.add(t, s),
    "maximum": lambda t, s: lucid.maximum(t, s),
    "inplace": lambda t, s: t.clone().sub_(s),
    "augmented": lambda t, s: operator.iadd(t.clone(), s),
}


def _same(fn: Callable[[object], object], scalar: np.generic) -> None:
    """``fn(scalar)`` is ``fn(scalar.item())``: the same tensor, or the same refusal."""
    try:
        want = fn(scalar.item())
    except (RuntimeError, ValueError, TypeError) as e:
        with pytest.raises(type(e)):
            fn(scalar)
        return
    got = fn(scalar)
    assert isinstance(got, lucid.Tensor) and isinstance(want, lucid.Tensor), type(got)
    assert got.dtype == want.dtype
    assert got.device == want.device
    np.testing.assert_array_equal(got.numpy(), want.numpy())


def _ids(values: list[object]) -> list[str]:
    return [f"{type(v).__name__}({v})" for v in values]


@pytest.mark.parametrize("op", _ARITHMETIC + _COMPARISON, ids=lambda f: f.__name__)
@pytest.mark.parametrize("scalar", _REAL, ids=_ids(_REAL))
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float16, lucid.int32])
def test_an_operator_takes_a_numpy_scalar_as_its_number(
    device: str, dtype: lucid.dtype, scalar: np.generic, op: Callable[..., object]
) -> None:
    t = lucid.tensor([1, 2, 5], dtype=dtype, device=device)
    _same(lambda s: op(t, s), scalar)


@pytest.mark.parametrize("op", _BITWISE, ids=lambda f: f.__name__)
@pytest.mark.parametrize("scalar", _INTEGER, ids=_ids(_INTEGER))
def test_a_bitwise_operator_takes_a_numpy_integer(
    device: str, scalar: np.generic, op: Callable[..., object]
) -> None:
    t = lucid.tensor([1, 6, 12], dtype=lucid.int32, device=device)
    _same(lambda s: op(t, s), scalar)


@pytest.mark.parametrize("spelling", list(_SPELLINGS))
@pytest.mark.parametrize("scalar", _REAL, ids=_ids(_REAL))
def test_every_spelling_takes_a_numpy_scalar(
    device: str, scalar: np.generic, spelling: str
) -> None:
    t = lucid.tensor([1.0, -2.0, 4.0], device=device)
    fn = _SPELLINGS[spelling]
    _same(lambda s: fn(t, s), scalar)


def test_a_complex_numpy_scalar_is_a_complex_number(device: str) -> None:
    t = lucid.tensor([1.0, 2.0], device=device)
    s = np.complex64(1.0 + 2.0j)
    _same(lambda v: t + v, s)
    _same(lambda v: t == v, s)


def test_equality_with_a_non_number_is_still_identity() -> None:
    t = lucid.tensor([1.0])
    assert operator.eq(t, None) is False
    assert operator.eq(t, "1") is False
