"""Unit tests for the P6 ``lucid.linalg`` completion: ``*_ex`` variants,
``lu``, ``ldl_solve``, and ``linalg.diagonal``.
"""

import numpy as np
import pytest

import lucid


def _spd(seed: int = 0) -> lucid.Tensor:
    """Symmetric positive-definite test matrix."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((4, 4)).astype(np.float32)
    A = (M @ M.T + 4 * np.eye(4)).astype(np.float32)
    return lucid.tensor(A.copy())


class TestCholeskyEx:
    def test_success_zero_info(self) -> None:
        A = _spd()
        L, info = lucid.linalg.cholesky_ex(A)
        assert int(info.item()) == 0
        recon = (L @ L.mT).numpy()
        np.testing.assert_allclose(recon, A.numpy(), atol=1e-3)

    def test_failure_returns_nonzero_info(self) -> None:
        # The all-zero matrix is not positive-definite — Cholesky should
        # fail and ``cholesky_ex`` should report a non-zero ``info``.
        zeroA = lucid.zeros(3, 3)
        L, info = lucid.linalg.cholesky_ex(zeroA)
        assert int(info.item()) != 0


class TestInvEx:
    def test_success(self) -> None:
        A = _spd()
        Ai, info = lucid.linalg.inv_ex(A)
        assert int(info.item()) == 0
        np.testing.assert_allclose((A @ Ai).numpy(), np.eye(4), atol=1e-3)

    def test_singular(self) -> None:
        S = lucid.zeros(3, 3)
        _, info = lucid.linalg.inv_ex(S)
        assert int(info.item()) != 0

    def test_check_errors_re_raises(self) -> None:
        with pytest.raises(Exception):
            lucid.linalg.inv_ex(lucid.zeros(3, 3), check_errors=True)


class TestSolveEx:
    def test_success(self) -> None:
        A = _spd()
        B = lucid.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
        X, info = lucid.linalg.solve_ex(A, B)
        assert int(info.item()) == 0
        np.testing.assert_allclose((A @ X).numpy(), B.numpy(), atol=1e-3)

    def test_left_false_rejected(self) -> None:
        A = _spd()
        B = lucid.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
        with pytest.raises(NotImplementedError):
            lucid.linalg.solve_ex(A, B, left=False)


class TestLu:
    def test_factor_product_recovers_A(self) -> None:
        A = lucid.tensor([[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]])
        P, L, U = lucid.linalg.lu(A)
        np.testing.assert_allclose((P @ L @ U).numpy(), A.numpy(), atol=1e-4)

    def test_L_is_unit_lower(self) -> None:
        P, L, U = lucid.linalg.lu(_spd())
        Lnp = L.numpy()
        # Diagonal of 1s.
        np.testing.assert_allclose(np.diag(Lnp), np.ones(4), atol=1e-6)
        # Strictly upper part is zero.
        for i in range(4):
            for j in range(i + 1, 4):
                assert abs(Lnp[i, j]) < 1e-6

    def test_U_is_upper(self) -> None:
        P, L, U = lucid.linalg.lu(_spd())
        Unp = U.numpy()
        for i in range(4):
            for j in range(i):
                assert abs(Unp[i, j]) < 1e-6

    def test_pivot_false_rejected(self) -> None:
        with pytest.raises(NotImplementedError):
            lucid.linalg.lu(_spd(), pivot=False)

    def test_non_square_rejected(self) -> None:
        with pytest.raises(ValueError):
            lucid.linalg.lu(lucid.zeros(3, 4))


class TestLdlSolve:
    def test_solves_simple_pivot_case(self) -> None:
        A = lucid.tensor([[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]])
        LD, piv = lucid.linalg.ldl_factor(A)
        B = lucid.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        X = lucid.linalg.ldl_solve(LD, piv, B)
        np.testing.assert_allclose((A @ X).numpy(), B.numpy(), atol=1e-3)


class TestLinalgDiagonal:
    def test_basic(self) -> None:
        A = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(lucid.linalg.diagonal(A).numpy(), [1.0, 5.0, 9.0])

    def test_offset(self) -> None:
        A = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(
            lucid.linalg.diagonal(A, offset=1).numpy(), [2.0, 6.0]
        )
        np.testing.assert_array_equal(
            lucid.linalg.diagonal(A, offset=-1).numpy(), [4.0, 8.0]
        )


class TestNamespacePolicy:
    def test_only_via_linalg(self) -> None:
        # H8: linalg ops have a single canonical path.  None of the
        # P6 additions should leak to top-level.
        for name in ("cholesky_ex", "inv_ex", "solve_ex", "lu", "ldl_solve"):
            assert not hasattr(lucid, name), (
                f"lucid.{name} should not exist at top level — H8 forbids "
                f"linalg shortcuts"
            )
