"""``complex128`` — the second complex dtype, and what it exposed.

It exists for :func:`lucid.linalg.eig`, whose output dtype is not a
choice: the eigenvalues of an ``f64`` matrix need 64-bit lanes, and
narrowing them to ``complex64`` would shed eight decimal digits.

Adding it turned two comments in the complex ops from true into false:

* ``complex_conj`` tested ``dt != Dtype::C64`` and called that branch
  "real dtypes", which those two were the same statement only while C64
  was the only complex type.  A ``complex128`` tensor took the identity
  branch and ``conj`` returned its argument unchanged.
* ``complex_combine`` read both halves as ``float*`` whatever they were,
  so ``complex([1., 2.], [3., 4.])`` on ``f64`` inputs answered
  ``[0+0j, 1.875+2.125j]`` — four plausible numbers assembled from the
  halves of four doubles.  ``f16`` was wrong differently.  Only ``f32``
  ever worked, and ``f32`` is what every test used.
"""

import io

import numpy as np
import pytest

import lucid


def _v(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.numpy())


Z128 = np.array([1 + 2j, 3 - 1j], dtype=np.complex128)
Z64 = Z128.astype(np.complex64)


# ── the dtype itself ──────────────────────────────────────────────────────────


def test_round_trips_through_numpy() -> None:
    z = lucid.tensor(Z128)
    assert z.dtype is lucid.complex128
    assert _v(z).dtype == np.complex128
    assert np.array_equal(_v(z), Z128)


def test_element_size_is_sixteen_bytes() -> None:
    assert lucid.complex128.itemsize == 16
    assert lucid.complex64.itemsize == 8


def test_is_complex_covers_both() -> None:
    assert lucid.tensor(Z128).is_complex()
    assert lucid.tensor(Z64).is_complex()
    assert not lucid.tensor(np.array([1.0])).is_complex()


def test_item_and_repr() -> None:
    assert lucid.tensor(np.array(1 + 2j)).item() == 1 + 2j
    assert "complex128" in repr(lucid.tensor(Z128))


def test_survives_a_checkpoint() -> None:
    buf = io.BytesIO()
    lucid.save({"z": lucid.tensor(Z128)}, buf)
    buf.seek(0)
    assert np.array_equal(_v(lucid.load(buf)["z"]), Z128)


# ── conj, which silently no-opped ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "arr,dtype", [(Z128, lucid.complex128), (Z64, lucid.complex64)]
)
def test_conj_negates_the_imaginary_part(arr, dtype) -> None:
    out = lucid.conj(lucid.tensor(arr))
    assert out.dtype is dtype
    assert np.allclose(_v(out), np.conj(arr))
    assert not np.allclose(_v(out), arr)  # the identity is the bug


def test_conj_of_a_real_tensor_is_still_the_identity() -> None:
    real = lucid.tensor(np.array([1.0, -2.0]))
    assert np.array_equal(_v(lucid.conj(real)), [1.0, -2.0])


# ── real / imag project the right lane width ──────────────────────────────────


@pytest.mark.parametrize("arr,lane", [(Z128, lucid.float64), (Z64, lucid.float32)])
def test_real_and_imag(arr, lane) -> None:
    z = lucid.tensor(arr)
    re, im = lucid.real(z), lucid.imag(z)
    assert re.dtype is lane and im.dtype is lane
    assert np.allclose(_v(re), arr.real)
    assert np.allclose(_v(im), arr.imag)


# ── complex(), which was wrong at three of four widths ────────────────────────


@pytest.mark.parametrize(
    "real_dtype,complex_dtype",
    [
        (lucid.float32, lucid.complex64),
        (lucid.float64, lucid.complex128),
        (lucid.float16, lucid.complex64),
        (lucid.bfloat16, lucid.complex64),
    ],
)
def test_complex_assembles_from_any_real_width(real_dtype, complex_dtype) -> None:
    """The half formats have no complex type of their own, so they widen."""
    re = lucid.tensor(np.array([1.0, 2.0]), dtype=real_dtype)
    im = lucid.tensor(np.array([3.0, 4.0]), dtype=real_dtype)
    out = lucid.complex(re, im)
    assert out.dtype is complex_dtype
    assert np.allclose(_v(out), [1 + 3j, 2 + 4j])


def test_the_parts_survive_a_round_trip() -> None:
    z = lucid.tensor(Z128)
    rebuilt = lucid.complex(lucid.real(z), lucid.imag(z))
    assert rebuilt.dtype is lucid.complex128
    assert np.array_equal(_v(rebuilt), Z128)


def test_complex_of_an_infinite_part_is_not_nan() -> None:
    """``re + 1j*im`` would make the real part ``im * 0`` — NaN when the
    imaginary part is infinite.  The parts are laid out, not computed."""
    out = lucid.complex(
        lucid.tensor(np.array([np.inf])), lucid.tensor(np.array([np.inf]))
    )
    assert _v(out)[0] == complex(np.inf, np.inf)
