"""``vander`` built its exponent row at a hardcoded dtype.

``pow`` requires both operands to agree, so an ``f64`` input raised
``DtypeMismatch (pow): expected float64, got float32`` — and ``f64`` is
what a tensor built from a Python list gets by default, so the plainest
possible call was the one that failed.

The device had exactly this bug and was fixed; the dtype one line over
was not.  A comment above the fix explains the device case and says
nothing about the dtype, which is how it survived.

The column order is *not* changed here.  NumPy's ``vander`` decreases by
default and the reference framework's increases; Lucid follows NumPy and
documents ``increasing=False``, so that is a stated convention rather
than a defect.  Both orders are checked against NumPy below.
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

VALUES = np.array([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (lucid.float32, "float32"),
        (lucid.float64, "float64"),
        (lucid.float16, "float32"),
        (lucid.bfloat16, "float32"),
    ],
)
def test_every_float_dtype_is_accepted(dtype, expected) -> None:
    """The half formats widen: ``arange`` has no half kernel, so the
    exponent cannot narrow to meet them."""
    out = lucid.linalg.vander(lucid.tensor(VALUES, dtype=dtype))
    assert expected in str(out.dtype)
    assert np.allclose(np.asarray(out.numpy()), np.vander(VALUES), atol=1e-2)


def test_the_default_list_input_works() -> None:
    """``lucid.tensor([...])`` is f64, which is precisely what raised."""
    out = lucid.linalg.vander(lucid.tensor([1.0, 2.0, 3.0]))
    assert np.allclose(np.asarray(out.numpy()), np.vander(np.array([1.0, 2.0, 3.0])))


@pytest.mark.parametrize("increasing", [False, True])
@pytest.mark.parametrize("n", [None, 2, 5])
def test_both_orders_match_numpy(increasing, n) -> None:
    out = lucid.linalg.vander(lucid.tensor(VALUES), N=n, increasing=increasing)
    expected = np.vander(VALUES, N=n, increasing=increasing)
    assert np.allclose(np.asarray(out.numpy()), expected)


def test_the_documented_default_is_decreasing() -> None:
    """Guard the convention, so a later 'fix' toward the reference has to
    be a deliberate API change rather than a quiet one."""
    out = np.asarray(lucid.linalg.vander(lucid.tensor([1.0, 2.0, 3.0])).numpy())
    assert np.allclose(out[1], [1.0, 2.0, 4.0][::-1])  # 4, 2, 1


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_the_exponent_follows_the_input_device_too() -> None:
    out = lucid.linalg.vander(lucid.tensor(VALUES, dtype=lucid.float32, device="metal"))
    assert "metal" in str(out.device)
    assert np.allclose(np.asarray(out.numpy()), np.vander(VALUES), atol=1e-4)
