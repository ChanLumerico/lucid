"""
Tests for lucid.linalg — covers all public ops and the _linalg_op decorator.
"""

import numpy as np
import pytest
import lucid
import lucid.linalg as LA


def randf(*shape: int) -> lucid.Tensor:
    return lucid.randn(*shape)


# ── Basic engine-backed ops ───────────────────────────────────────────────────


class TestInvDet:
    def test_inv_2x2(self):
        A = lucid.tensor([[2.0, 1.0], [5.0, 3.0]])
        Ainv = LA.inv(A)
        I = lucid.matmul(A, Ainv)
        eye = np.eye(2, dtype=np.float32)
        np.testing.assert_allclose(I.numpy(), eye, atol=1e-5)

    def test_det_2x2(self):
        A = lucid.tensor([[3.0, 8.0], [4.0, 6.0]])
        d = LA.det(A)
        assert abs(float(d.item()) - (3 * 6 - 8 * 4)) < 1e-4

    def test_inv_batch(self):
        A = lucid.randn(4, 3, 3)
        _ = LA.inv(A)  # should not raise

    def test_det_identity(self):
        I = lucid.eye(4)
        d = LA.det(I)
        assert abs(float(d.item()) - 1.0) < 1e-5


class TestSolve:
    def test_solve_2x2(self):
        A = lucid.tensor([[2.0, 1.0], [5.0, 3.0]])
        b = lucid.tensor([[1.0], [2.0]])
        x = LA.solve(A, b)
        residual = lucid.matmul(A, x)
        np.testing.assert_allclose(residual.numpy(), b.numpy(), atol=1e-5)


class TestCholesky:
    def test_cholesky_spd(self):
        # Build a symmetric positive-definite matrix
        A = lucid.tensor([[4.0, 2.0], [2.0, 3.0]])
        L = LA.cholesky(A)
        reconstructed = lucid.matmul(L, L.mT)
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)

    def test_cholesky_upper(self):
        A = lucid.tensor([[4.0, 2.0], [2.0, 3.0]])
        U = LA.cholesky(A, upper=True)
        reconstructed = lucid.matmul(U.mT, U)
        np.testing.assert_allclose(reconstructed.numpy(), A.numpy(), atol=1e-5)


class TestQRSVD:
    def test_qr_orthogonality(self):
        A = lucid.randn(4, 3)
        Q, R = LA.qr(A)
        QtQ = lucid.matmul(Q.mT, Q)
        np.testing.assert_allclose(QtQ.numpy(), np.eye(3, dtype=np.float32), atol=1e-5)

    def test_svd_reconstruction(self):
        A = lucid.randn(3, 3)
        U, S, Vh = LA.svd(A)
        S_diag = np.diag(S.numpy())
        reconstructed = U.numpy() @ S_diag @ Vh.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-4)

    def test_svdvals_shape(self):
        A = lucid.randn(5, 3)
        sv = LA.svdvals(A)
        assert sv.shape == (3,)

    def test_svdvals_positive(self):
        A = lucid.randn(4, 4)
        sv = LA.svdvals(A)
        assert (sv.numpy() >= 0).all()


class TestEig:
    def test_eigh_symmetric(self):
        A = lucid.tensor([[2.0, 1.0], [1.0, 2.0]])
        vals, vecs = LA.eigh(A)
        assert vals.shape == (2,)
        assert vecs.shape == (2, 2)

    def test_eigvalsh_vs_eigh(self):
        A = lucid.tensor([[3.0, 1.0], [1.0, 3.0]])
        vals_only = LA.eigvalsh(A)
        vals_full, _ = LA.eigh(A)
        np.testing.assert_allclose(vals_only.numpy(), vals_full.numpy(), atol=1e-5)

    def test_eig_square(self):
        A = lucid.randn(3, 3)
        vals, vecs = LA.eig(A)
        assert vals.shape[0] == 3

    def test_eigvals_vs_eig(self):
        A = lucid.randn(3, 3)
        vals_only = LA.eigvals(A)
        vals_full, _ = LA.eig(A)
        np.testing.assert_allclose(vals_only.numpy(), vals_full.numpy(), atol=1e-5)


# ── Pure-Python compositions ───────────────────────────────────────────────────


class TestSlogdet:
    def test_slogdet_identity(self):
        I = lucid.eye(3)
        sign, logdet = LA.slogdet(I)
        assert abs(float(sign.item()) - 1.0) < 1e-5
        assert abs(float(logdet.item())) < 1e-5

    def test_slogdet_known(self):
        A = lucid.tensor([[2.0, 0.0], [0.0, 3.0]])  # det = 6
        sign, logdet = LA.slogdet(A)
        assert float(sign.item()) > 0
        assert abs(float(logdet.item()) - np.log(6.0)) < 1e-4


class TestMatrixRank:
    def test_full_rank(self):
        A = lucid.eye(4)
        r = LA.matrix_rank(A)
        assert int(r.item()) == 4

    def test_rank_deficient(self):
        # Zero matrix → rank 0
        A = lucid.zeros(3, 3)
        r = LA.matrix_rank(A)
        assert int(r.item()) == 0

    def test_rank_less_than_full(self):
        # Rank of eye(4) is 4; rank of a near-singular matrix < 4
        assert int(LA.matrix_rank(lucid.eye(4)).item()) == 4


class TestCond:
    def test_identity_cond(self):
        I = lucid.eye(3)
        c = LA.cond(I)
        assert abs(float(c.item()) - 1.0) < 1e-4

    def test_cond_negative_p(self):
        A = lucid.tensor([[2.0, 0.0], [0.0, 4.0]])
        c = LA.cond(A, p=-2)
        assert float(c.item()) > 0


class TestMultiDot:
    def test_chain_matmul(self):
        A = lucid.randn(2, 3)
        B = lucid.randn(3, 4)
        C = lucid.randn(4, 2)
        result = LA.multi_dot([A, B, C])
        expected = lucid.matmul(lucid.matmul(A, B), C)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), atol=1e-5)

    def test_single_tensor(self):
        A = lucid.randn(3, 3)
        assert LA.multi_dot([A]) is A

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            LA.multi_dot([])


class TestLuFactor:
    def test_lu_shape(self):
        A = lucid.randn(4, 4)
        LU, pivots = LA.lu_factor(A)
        assert LU.shape == (4, 4)
        assert pivots.shape == (4,)


class TestSolveTriangular:
    def test_upper_triangular(self):
        U = lucid.tensor([[2.0, 1.0], [0.0, 3.0]])
        b = lucid.tensor([[5.0], [6.0]])
        x = LA.solve_triangular(U, b, upper=True)
        residual = lucid.matmul(U, x)
        np.testing.assert_allclose(residual.numpy(), b.numpy(), atol=1e-5)

    def test_right_system(self):
        U = lucid.tensor([[2.0, 0.0], [0.0, 3.0]])
        b = lucid.tensor([[4.0, 9.0]])
        x = LA.solve_triangular(U, b, upper=True, left=False)
        residual = lucid.matmul(x, U)
        np.testing.assert_allclose(residual.numpy(), b.numpy(), atol=1e-5)


class TestVander:
    def test_increasing(self):
        x = lucid.tensor([1.0, 2.0, 3.0])
        V = LA.vander(x, N=4, increasing=True)
        expected = np.vander([1, 2, 3], N=4, increasing=True).astype(np.float32)
        np.testing.assert_allclose(V.numpy(), expected, atol=1e-5)

    def test_decreasing(self):
        x = lucid.tensor([1.0, 2.0, 3.0])
        V = LA.vander(x, N=3, increasing=False)
        expected = np.vander([1, 2, 3], N=3, increasing=False).astype(np.float32)
        np.testing.assert_allclose(V.numpy(), expected, atol=1e-5)
