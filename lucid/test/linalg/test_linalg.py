"""Tests for lucid.linalg operations."""

import pytest
import numpy as np
import lucid
import lucid.linalg as LA
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


def _spd(n, seed=0):
    """Return a symmetric positive-definite n×n matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float32)
    return lucid.tensor((A @ A.T + np.eye(n) * n).astype(np.float32))


def _sq(n, seed=0):
    """Return a random n×n matrix (likely full rank)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float32) * 0.5 + np.eye(n) * 1.5
    return lucid.tensor(A.astype(np.float32))


class TestInv:
    def test_inverse_identity(self):
        I = lucid.eye(4)
        assert_close(LA.inv(I), I, atol=1e-5)

    def test_inv_times_A_is_I(self):
        A = _sq(4)
        Ainv = LA.inv(A)
        prod = lucid.matmul(A, Ainv)
        assert_close(prod, lucid.eye(4), atol=1e-4)

    def test_output_shape(self):
        A = _sq(5)
        assert LA.inv(A).shape == (5, 5)


class TestDet:
    def test_det_identity(self):
        I = lucid.eye(4)
        d = LA.det(I)
        assert abs(float(d.item()) - 1.0) < 1e-4

    def test_det_zero_for_singular(self):
        # Rank-deficient matrix
        A = lucid.zeros(3, 3)
        d = LA.det(A)
        assert abs(float(d.item())) < 1e-4

    def test_det_sign(self):
        # Permutation matrix swaps two rows → det = -1
        P = lucid.tensor([[0.0, 1.0], [1.0, 0.0]])
        d = LA.det(P)
        assert abs(float(d.item()) + 1.0) < 1e-4


class TestSolve:
    def test_solve_identity_rhs(self):
        I = lucid.eye(4)
        b = make_tensor((4, 2))
        x = LA.solve(I, b)
        assert_close(x, b, atol=1e-5)

    def test_Ax_equals_b(self):
        A = _sq(4)
        b = make_tensor((4, 2))
        x = LA.solve(A, b)
        residual = lucid.matmul(A, x) - b
        assert float(lucid.max(lucid.abs(residual)).item()) < 1e-3


class TestQR:
    def test_output_shapes(self):
        A = make_tensor((6, 4))
        Q, R = LA.qr(A)
        assert Q.shape == (6, 4)
        assert R.shape == (4, 4)

    def test_QR_reconstruction(self):
        A = make_tensor((4, 4))
        Q, R = LA.qr(A)
        assert_close(lucid.matmul(Q, R), A, atol=1e-4)

    def test_Q_orthonormal(self):
        A = make_tensor((4, 4))
        Q, _ = LA.qr(A)
        # Q^T @ Q ≈ I
        prod = lucid.matmul(lucid.transpose(Q), Q)
        assert_close(prod, lucid.eye(4), atol=1e-4)


class TestSVD:
    def test_output_shapes(self):
        A = make_tensor((4, 6))
        U, S, Vh = LA.svd(A)
        assert U.shape[0] == 4
        assert S.shape[0] == min(4, 6)

    def test_singular_values_non_negative(self):
        A = make_tensor((4, 4))
        _, S, _ = LA.svd(A)
        assert (S.numpy() >= -1e-5).all()

    def test_reconstruction(self):
        A = make_tensor((3, 3), seed=42)
        U, S, Vh = LA.svd(A)
        # Reconstruct: U @ diag(S) @ Vh
        S_diag = lucid.tensor(np.diag(S.numpy()))
        Ar = lucid.matmul(lucid.matmul(U, S_diag), Vh)
        assert_close(Ar, A, atol=1e-4)


class TestCholesky:
    def test_lower_triangular(self):
        A = _spd(4)
        L = LA.cholesky(A)
        # Upper triangle should be zero
        arr = L.numpy()
        assert np.allclose(np.triu(arr, 1), 0, atol=1e-5)

    def test_LLT_reconstruction(self):
        A = _spd(4)
        L = LA.cholesky(A)
        Ar = lucid.matmul(L, lucid.transpose(L))
        assert_close(Ar, A, atol=1e-4)


class TestEigh:
    def test_eigenvalue_shapes(self):
        A = _spd(4)
        vals, vecs = LA.eigh(A)
        assert vals.shape == (4,)
        assert vecs.shape == (4, 4)

    def test_eigenvalues_non_negative(self):
        A = _spd(4)
        vals, _ = LA.eigh(A)
        assert (vals.numpy() >= -1e-4).all()

    def test_reconstruction(self):
        A = _spd(4)
        vals, vecs = LA.eigh(A)
        # A ≈ V @ diag(λ) @ V^T
        D = lucid.tensor(np.diag(vals.numpy()))
        Ar = lucid.matmul(lucid.matmul(vecs, D), lucid.transpose(vecs))
        assert_close(Ar, A, atol=1e-3)


class TestNorm:
    def test_frobenius(self):
        A = lucid.eye(3)
        norm = LA.matrix_norm(A, ord="fro")
        # ||I_3||_F = sqrt(3)
        assert abs(float(norm.item()) - np.sqrt(3)) < 1e-4

    def test_vector_norm_l2(self):
        x = lucid.tensor([3.0, 4.0])
        assert abs(float(LA.vector_norm(x).item()) - 5.0) < 1e-4

    def test_norm_positive(self):
        A = make_tensor((3, 3))
        assert float(LA.norm(A).item()) >= 0.0


class TestPinv:
    def test_pinv_square_invertible(self):
        A = _sq(3)
        Ap = LA.pinv(A)
        # A @ Ap @ A ≈ A
        result = lucid.matmul(lucid.matmul(A, Ap), A)
        assert_close(result, A, atol=1e-3)

    def test_pinv_shape(self):
        A = make_tensor((4, 3))
        Ap = LA.pinv(A)
        assert Ap.shape == (3, 4)


class TestMatrixPower:
    def test_power_zero_is_identity(self):
        A = _sq(3)
        result = LA.matrix_power(A, 0)
        assert_close(result, lucid.eye(3), atol=1e-5)

    def test_power_one_is_identity_op(self):
        A = _sq(3)
        result = LA.matrix_power(A, 1)
        assert_close(result, A, atol=1e-5)

    def test_power_two(self):
        A = _sq(3)
        result = LA.matrix_power(A, 2)
        expected = lucid.matmul(A, A)
        assert_close(result, expected, atol=1e-4)


class TestLinalg:
    def test_slogdet(self):
        A = _sq(3)
        sign, logdet = LA.slogdet(A)
        # exp(logdet) * sign ≈ det(A)
        det_val = LA.det(A)
        reconstructed = sign.item() * np.exp(float(logdet.item()))
        assert abs(reconstructed - float(det_val.item())) < 1e-3

    def test_cross_1d(self):
        a = lucid.tensor([1.0, 0.0, 0.0])
        b = lucid.tensor([0.0, 1.0, 0.0])
        c = LA.cross(a, b)
        assert_close(c, lucid.tensor([0.0, 0.0, 1.0]))

    def test_cross_2d(self):
        a = lucid.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = lucid.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        c = LA.cross(a, b, dim=1)
        assert c.shape == (2, 3)

    def test_lstsq(self):
        # Solve overdetermined system; returns (solution, ...) tuple
        A = make_tensor((6, 3))
        b = make_tensor((6, 1))
        result = LA.lstsq(A, b)
        solution = result[0] if isinstance(result, tuple) else result
        assert solution.ndim >= 1 and solution.shape[0] == 3
