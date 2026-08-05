"""``lu_factor`` accepts any shape, because ``?getrf`` always did.

LU with partial pivoting is defined for an m-by-n matrix: ``P A = L U``
with ``L`` unit-lower-trapezoidal ``m × k`` and ``U`` upper-trapezoidal
``k × n``, where ``k = min(m, n)``.  This op required a square input,
which was a restriction of the wrapper and not of the factorisation —
the LAPACK call underneath was passing the same extent for both
dimensions, so the rectangular case was unreachable rather than
unsupported.

The pivot vector has ``min(m, n)`` entries, one per elimination step,
and there are only as many steps as the shorter side.  ``n`` was right
only because nothing but a square matrix could get in.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as L

RECTANGULAR = [(3, 4), (4, 3), (1, 5), (5, 1), (2, 7), (7, 2)]


def _factor(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lu, piv = L.lu_factor(lucid.tensor(a))
    return np.asarray(lu.numpy()), np.asarray(piv.numpy())


def _reconstruct(a: np.ndarray, lu: np.ndarray, piv: np.ndarray) -> np.ndarray:
    """Undo the packing and the pivots, giving back ``A``."""
    m, n = a.shape
    k = min(m, n)
    lower = np.tril(lu[:, :k], -1) + np.eye(m, k)
    upper = np.triu(lu[:k, :])
    out = lower @ upper
    for i in range(k - 1, -1, -1):  # pivots unwind last-applied-first
        j = piv[i] - 1  # LAPACK's indices are 1-based
        if i != j:
            out[[i, j]] = out[[j, i]]
    return out


# ── shapes ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", RECTANGULAR + [(3, 3), (6, 6)])
def test_output_shapes(shape) -> None:
    a = np.random.default_rng(sum(shape)).standard_normal(shape)
    lu, piv = _factor(a)
    assert lu.shape == shape
    assert piv.shape == (min(shape),)


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 4, 3), (2, 3, 5, 2)])
def test_batches_carry_through(shape) -> None:
    a = np.random.default_rng(len(shape)).standard_normal(shape)
    lu, piv = _factor(a)
    assert lu.shape == shape
    assert piv.shape == (*shape[:-2], min(shape[-2:]))


def test_pivots_are_int32() -> None:
    _, piv = L.lu_factor(lucid.tensor(np.random.default_rng(0).standard_normal((3, 5))))
    assert piv.dtype is lucid.int32


# ── the property the factorisation exists for ─────────────────────────────────


@pytest.mark.parametrize("shape", RECTANGULAR + [(4, 4)])
def test_p_l_u_reconstructs_the_matrix(shape) -> None:
    a = np.random.default_rng(sum(shape) + 7).standard_normal(shape)
    lu, piv = _factor(a)
    assert np.allclose(_reconstruct(a, lu, piv), a, atol=1e-12)


@pytest.mark.parametrize("shape", RECTANGULAR)
def test_the_factors_have_the_trapezoidal_shapes(shape) -> None:
    """``L`` is ``m × k`` and ``U`` is ``k × n`` — not both square."""
    m, n = shape
    k = min(m, n)
    a = np.random.default_rng(m * n).standard_normal(shape)
    lu, _ = _factor(a)
    lower = np.tril(lu[:, :k], -1) + np.eye(m, k)
    upper = np.triu(lu[:k, :])
    assert lower.shape == (m, k)
    assert upper.shape == (k, n)
    assert np.allclose(np.diag(lower)[:k], 1.0)  # unit diagonal
    assert np.allclose(np.tril(upper, -1), 0.0)  # upper-trapezoidal


def test_a_singular_rectangular_matrix_still_factors() -> None:
    """``getrf`` reports a zero pivot through info > 0; a rank-deficient
    rectangular matrix is ordinary input, not an error."""
    a = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])  # rank 1
    lu, piv = _factor(a)
    assert lu.shape == (2, 3)
    assert np.allclose(_reconstruct(a, lu, piv), a, atol=1e-12)


# ── the degenerate shapes still work ──────────────────────────────────────────


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
def test_empty(shape) -> None:
    lu, piv = L.lu_factor(lucid.tensor(np.zeros(shape)))
    assert tuple(lu.shape) == shape
    assert tuple(piv.shape) == (0,)


# ── lu_solve still needs a square factor ──────────────────────────────────────


def test_lu_solve_refuses_a_rectangular_factor() -> None:
    """Only a square system has a solution.  Handed a rectangular factor,
    ``getrs`` read ``n`` rows from a matrix with fewer and answered
    ``[nan, nan, -inf]``."""
    a = np.random.default_rng(0).standard_normal((3, 4))
    lu, piv = L.lu_factor(lucid.tensor(a))
    with pytest.raises(Exception, match="square"):
        L.lu_solve(lu, piv, lucid.tensor(np.ones((3, 1))))


def test_lu_solve_on_a_square_factor_is_unchanged() -> None:
    a = np.random.default_rng(3).standard_normal((4, 4))
    b = np.random.default_rng(4).standard_normal((4, 2))
    lu, piv = L.lu_factor(lucid.tensor(a))
    x = np.asarray(L.lu_solve(lu, piv, lucid.tensor(b)).numpy())
    assert np.allclose(a @ x, b, atol=1e-10)
