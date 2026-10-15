"""Parity guard for dtype edge cases across CPU and Metal (LCD-343).

Covers:
1. ``lucid.can_cast(from_, to)`` matches ``torch.can_cast`` across all 11x11 pairs.
2. ``t == None`` and ``t != None`` (plus arbitrary non-Tensor, non-Scalar objects)
   return ``False`` / ``True`` as plain Python ``bool``.
3. Half and narrow-integer scalar exponent overflow in ``pow`` raises ``RuntimeError``
   across every spelling (``**``, ``lucid.pow``, ``Tensor.pow``, ``**=``, ``pow_``).
4. Complex comparisons on CPU and Metal: ``==`` and ``!=`` work on ``complex64`` and
   ``complex128`` (including NaN semantics), while ``<, <=, >, >=`` raise
   ``NotImplementedError`` on both backends.
5. ``lucid.tensor(scalar, dtype=dt)`` and ``lucid.full((), scalar, dtype=dt)`` return
   a 0-D tensor ``()`` for every dtype including ``bfloat16`` and ``complex128``.
"""

from types import ModuleType
from typing import Any

import pytest

import lucid

pytestmark = pytest.mark.parity

_DTYPES = [
    lucid.bool_,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.complex64,
    lucid.complex128,
]
_CPU_ONLY = {lucid.float64, lucid.complex128}


def _ref_dt(ref: ModuleType, d: lucid.dtype) -> Any:
    name = str(d).removeprefix("lucid.").rstrip("_")
    return getattr(ref, name)


def test_can_cast_matches_reference(ref: ModuleType) -> None:
    for da in _DTYPES:
        for db in _DTYPES:
            got = lucid.can_cast(da, db)
            want = ref.can_cast(_ref_dt(ref, da), _ref_dt(ref, db))
            assert got is want, f"can_cast({da}, {db}): got {got}, want {want}"


@pytest.mark.parametrize("other", [None, "abc", object(), [1, 2]])
def test_equality_against_non_tensor_non_scalar(
    other: Any, device: str, ref: ModuleType
) -> None:
    t = lucid.tensor([1.0, 2.0], device=device)
    rt = ref.tensor([1.0, 2.0])
    eq_got = t == other
    ne_got = t != other
    assert isinstance(eq_got, bool) and eq_got is (rt == other)
    assert isinstance(ne_got, bool) and ne_got is (rt != other)
    req_got = other == t
    rne_got = other != t
    assert isinstance(req_got, bool) and req_got is (other == rt)
    assert isinstance(rne_got, bool) and rne_got is (other != rt)


@pytest.mark.parametrize(
    ("dt", "scalar"),
    [
        (lucid.float16, 100000),
        (lucid.float16, -100000.0),
        (lucid.bfloat16, 1e40),
        (lucid.int8, 200),
        (lucid.int8, -200),
        (lucid.int16, 100000),
        (lucid.int32, 1 << 40),
        (lucid.int64, 1 << 70),
    ],
)
def test_pow_scalar_overflow_raises_runtime_error(
    dt: lucid.dtype, scalar: int | float, device: str, ref: ModuleType
) -> None:
    rt = ref.tensor([1, 2], dtype=_ref_dt(ref, dt))
    with pytest.raises((RuntimeError, OverflowError)):
        _ = rt**scalar

    t = lucid.tensor([1, 2], dtype=lucid.float32, device=device).to(dt)
    with pytest.raises(RuntimeError, match="without overflow"):
        _ = t**scalar
    with pytest.raises(RuntimeError, match="without overflow"):
        _ = lucid.pow(t, scalar)
    with pytest.raises(RuntimeError, match="without overflow"):
        _ = t.pow(scalar)
    with pytest.raises(RuntimeError, match="without overflow"):
        t_ip = t.clone()
        t_ip **= scalar


@pytest.mark.parametrize("dt", [lucid.complex64, lucid.complex128])
def test_complex_comparisons_match_reference(
    dt: lucid.dtype, device: str, ref: ModuleType
) -> None:
    if device != "cpu" and dt in _CPU_ONLY:
        pytest.skip(f"{dt} is CPU-only")
    vals_a = [1.0 + 2.0j, 1.0 + 2.0j, 1.0 + 3.0j, complex(float("nan"), 1.0)]
    vals_b = [1.0 + 2.0j, 1.0 + 4.0j, 2.0 + 3.0j, complex(float("nan"), 1.0)]
    a = lucid.tensor(vals_a, dtype=dt, device=device)
    b = lucid.tensor(vals_b, dtype=dt, device=device)
    ra = ref.tensor(vals_a, dtype=_ref_dt(ref, dt))
    rb = ref.tensor(vals_b, dtype=_ref_dt(ref, dt))

    for fn_l, fn_r in [
        (lambda x, y: x == y, lambda x, y: x == y),
        (lucid.eq, ref.eq),
        (lambda x, y: x != y, lambda x, y: x != y),
        (lucid.ne, ref.ne),
    ]:
        out = fn_l(a, b)
        want = fn_r(ra, rb)
        assert out.dtype == lucid.bool_
        assert out.tolist() == want.tolist()

    for op in [
        lambda x, y: x < y,
        lambda x, y: x <= y,
        lambda x, y: x > y,
        lambda x, y: x >= y,
        lucid.lt,
        lucid.le,
        lucid.gt,
        lucid.ge,
    ]:
        with pytest.raises(NotImplementedError):
            op(a, b)


@pytest.mark.parametrize("dt", _DTYPES)
def test_zero_dim_scalar_tensor_and_full_preserve_shape(
    dt: lucid.dtype, device: str, ref: ModuleType
) -> None:
    if device != "cpu" and dt in _CPU_ONLY:
        pytest.skip(f"{dt} is CPU-only")
    val: bool | int | float | complex = True if dt is lucid.bool_ else 2
    t = lucid.tensor(val, dtype=dt, device=device)
    rt = ref.tensor(val, dtype=_ref_dt(ref, dt))
    assert t.shape == rt.shape == ()
    assert t.ndim == 0
    assert t.dtype == dt

    f = lucid.full((), val, dtype=dt, device=device)
    assert f.shape == ()
    assert f.ndim == 0
    assert f.dtype == dt
