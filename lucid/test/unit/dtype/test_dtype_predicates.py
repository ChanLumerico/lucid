"""The free dtype predicates and the Tensor methods must answer alike.

``lucid.is_floating_point`` / ``is_complex`` / ``is_signed`` were three
hand-written frozensets of engine enums.  ``bfloat16`` and ``complex128``
later became engine dtypes of their own, were added to the *methods*
(``Tensor.is_floating_point`` / ``Tensor.is_complex``) and not to the
sets — so the free function and the method disagreed on exactly those
two, and the disagreement was load-bearing: ``lucid.autograd`` gates
differentiability on the free function, so ``jacobian`` refused a
bfloat16 input for being non-floating-point.

The sets are now derived from the dtype registry.  These tests are the
guard that says so: a dtype added to ``_ENGINE_TO_DTYPE`` without a kind
the derivation understands, or a method that starts disagreeing with its
free function, fails here rather than in someone's gradient.
"""

import pytest

import lucid

_ALL_DTYPES = [
    lucid.float16,
    lucid.bfloat16,
    lucid.float32,
    lucid.float64,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.bool_,
    lucid.complex64,
    lucid.complex128,
]

_FLOATING = {lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64}
_COMPLEX = {lucid.complex64, lucid.complex128}
_UNSIGNED = {lucid.bool_}


@pytest.mark.parametrize("dt", _ALL_DTYPES, ids=lambda d: d._name)
def test_free_function_agrees_with_method(dt: lucid.dtype) -> None:
    x = lucid.zeros(2, dtype=dt)
    assert lucid.is_floating_point(x) == x.is_floating_point()
    assert lucid.is_complex(x) == x.is_complex()


@pytest.mark.parametrize("dt", _ALL_DTYPES, ids=lambda d: d._name)
def test_is_floating_point(dt: lucid.dtype) -> None:
    x = lucid.zeros(2, dtype=dt)
    assert lucid.is_floating_point(x) is (dt in _FLOATING)


@pytest.mark.parametrize("dt", _ALL_DTYPES, ids=lambda d: d._name)
def test_is_complex(dt: lucid.dtype) -> None:
    x = lucid.zeros(2, dtype=dt)
    assert lucid.is_complex(x) is (dt in _COMPLEX)


@pytest.mark.parametrize("dt", _ALL_DTYPES, ids=lambda d: d._name)
def test_is_signed(dt: lucid.dtype) -> None:
    # Everything numeric is signed here; only bool is not.
    x = lucid.zeros(2, dtype=dt)
    assert lucid.is_signed(x) is (dt not in _UNSIGNED)


def test_every_registered_dtype_is_covered() -> None:
    # The point of deriving the sets from the registry is that a new
    # dtype cannot slip past.  If one is added, this list must grow too.
    from lucid._dtype import _ENGINE_TO_DTYPE

    assert set(_ENGINE_TO_DTYPE.values()) == set(_ALL_DTYPES)


def test_bfloat16_can_be_differentiated() -> None:
    # The concrete downstream failure: the differentiability gate reads
    # the free function, so a False here refused a legitimate input.
    from lucid.autograd._functional import _require_differentiable

    _require_differentiable(lucid.zeros(2, dtype=lucid.bfloat16), "jacobian")


@pytest.mark.parametrize("dt", [lucid.int64, lucid.complex64, lucid.bool_])
def test_non_floating_is_still_refused(dt: lucid.dtype) -> None:
    from lucid.autograd._functional import _require_differentiable

    with pytest.raises(TypeError, match="floating-point"):
        _require_differentiable(lucid.zeros(2, dtype=dt), "jacobian")
