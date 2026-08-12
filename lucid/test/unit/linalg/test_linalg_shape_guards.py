"""The CPU linalg backend must refuse the shapes it cannot compute.

The backend reads ``m`` and ``n`` off the last two dimensions of ``H``
and loops over neither a batch nor a missing axis, so two malformed
inputs used to be *answered* rather than refused — which is worse than
an error, because the answer has the right dtype and a plausible shape:

    H (3, 4, 3), tau (3, 3)  ->  Q (4, 3)   the other two matrices gone
    H (4,),      tau (4,)    ->  Q (0, 0)   an empty tensor, no complaint

``lstsq`` has the same shape of hole one function over, and there it
is not only a wrong answer: it copies ``m`` rows out of ``B`` without
asking how many ``B`` has, which segfaulted mid-``memmove`` inside the
audit's grad2 axis and returned a plausible answer on the runs where the
over-read happened to land on mapped memory.

``qr`` is unaffected — it loops the batch itself — so these are about the
direct entry points, the ones reached for when the caller already holds
reflectors or an over-determined system.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as LA


def _matrix(m: int, n: int) -> lucid.Tensor:
    lucid.manual_seed(0)
    return lucid.randn(m, n, dtype=lucid.float64)


def test_a_batched_H_is_refused_rather_than_silently_truncated() -> None:
    H = lucid.randn(3, 4, 3, dtype=lucid.float64)
    tau = lucid.randn(3, 3, dtype=lucid.float64)
    with pytest.raises(ValueError, match="single 2-D matrix"):
        LA.householder_product(H, tau)


def test_a_one_dimensional_H_is_refused_rather_than_returning_empty() -> None:
    H = lucid.randn(4, dtype=lucid.float64)
    tau = lucid.randn(4, dtype=lucid.float64)
    with pytest.raises(ValueError, match="single 2-D matrix"):
        LA.householder_product(H, tau)


def test_a_non_vector_tau_is_refused() -> None:
    with pytest.raises(ValueError, match="tau must be 1-D"):
        LA.householder_product(_matrix(4, 3), lucid.randn(2, 3, dtype=lucid.float64))


def test_too_few_reflectors_are_refused() -> None:
    """``tau`` shorter than ``min(m, n)`` is an over-read waiting to happen."""
    with pytest.raises(ValueError, match="at least min"):
        LA.householder_product(_matrix(4, 3), lucid.randn(2, dtype=lucid.float64))


@pytest.mark.parametrize(("m", "n"), [(4, 3), (3, 3), (5, 2), (2, 5)])
def test_a_well_formed_call_still_works(m: int, n: int) -> None:
    """The guard must not cost the supported case.

    Lucid exposes no raw-reflector factorisation, so the reflectors here
    are synthetic — which is fine, because what is being pinned is that
    a well-formed *shape* reaches the backend and comes back with the
    documented ``(m, min(m, n))`` and finite values.  Whether those
    particular numbers spell an orthogonal matrix is ``qr``'s question,
    and the last test asks it.
    """
    lucid.manual_seed(0)
    k = min(m, n)
    q = LA.householder_product(
        lucid.randn(m, n, dtype=lucid.float64),
        lucid.randn(k, dtype=lucid.float64) * 0.1,
    )
    assert tuple(q.shape) == (m, k)
    assert np.isfinite(q.numpy()).all()


def test_a_longer_tau_is_accepted() -> None:
    """Only *too few* reflectors are an error; extras are ignored by LAPACK."""
    lucid.manual_seed(0)
    q = LA.householder_product(
        lucid.randn(4, 3, dtype=lucid.float64),
        lucid.randn(3, dtype=lucid.float64) * 0.1,
    )
    assert tuple(q.shape) == (4, 3)


# ── lstsq, the same pattern one function over ────────────────────────────────


def test_lstsq_refuses_a_one_dimensional_A() -> None:
    """``shape[-2]`` on a rank-1 shape underflowed into a 1.8e19-byte ask."""
    with pytest.raises(ValueError, match="single 2-D matrix"):
        LA.lstsq(
            lucid.randn(4, dtype=lucid.float64), lucid.randn(4, dtype=lucid.float64)
        )


def test_lstsq_refuses_a_batched_A() -> None:
    with pytest.raises(ValueError, match="single 2-D matrix"):
        LA.lstsq(
            lucid.randn(3, 4, 3, dtype=lucid.float64),
            lucid.randn(3, 4, 2, dtype=lucid.float64),
        )


def test_lstsq_refuses_a_B_with_too_few_rows() -> None:
    """The over-read itself: ``m`` rows copied out of a shorter buffer.

    This is the one that segfaulted mid-``memmove`` — and it had been
    *returning an answer* on the runs where the read happened to land on
    mapped memory, which is the worse half of the bug.
    """
    with pytest.raises(ValueError, match="one row per row of A"):
        LA.lstsq(
            lucid.randn(6, 3, dtype=lucid.float64),
            lucid.randn(2, 2, dtype=lucid.float64),
        )


@pytest.mark.parametrize("b_shape", [(3,), (3, 1), (3, 4)])
def test_lstsq_still_solves_what_it_should(b_shape: tuple[int, ...]) -> None:
    lucid.manual_seed(0)
    a = lucid.randn(3, 2, dtype=lucid.float64)
    sol, *_ = LA.lstsq(a, lucid.randn(*b_shape, dtype=lucid.float64))
    assert tuple(sol.shape)[0] == 2
    assert np.isfinite(sol.numpy()).all()


def test_lstsq_matches_its_documented_example() -> None:
    a = lucid.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b = lucid.tensor([[6.0], [9.0], [12.0]])
    sol, *_ = LA.lstsq(a, b)
    np.testing.assert_allclose(sol.numpy().ravel(), [3.0, 3.0], rtol=1e-6, atol=1e-6)


def test_qr_still_handles_a_batch() -> None:
    """The batch path users actually have, and the reason the guard points there."""
    lucid.manual_seed(0)
    a = lucid.randn(3, 4, 3, dtype=lucid.float64)
    q, r = LA.qr(a)
    assert tuple(q.shape) == (3, 4, 3)
    recon = (q @ r).numpy()
    np.testing.assert_allclose(recon, a.numpy(), rtol=1e-10, atol=1e-10)
