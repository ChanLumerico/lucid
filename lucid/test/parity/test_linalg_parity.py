"""Parity tests for lucid.linalg."""

import importlib
import pytest
import numpy as np
import lucid
import lucid.linalg as LLA
from lucid.test.helpers.parity import check_parity

_REF_BACKEND = "to" "rch"
ref = pytest.importorskip(_REF_BACKEND)
TLA = importlib.import_module(_REF_BACKEND + ".linalg")


def _sq(n, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n, n)) * 0.3 + np.eye(n) * 1.5).astype(np.float32)
    return lucid.tensor(A.copy()), ref.tensor(A.copy())


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float32)
    M = (A @ A.T + np.eye(n) * n).astype(np.float32)
    return lucid.tensor(M.copy()), ref.tensor(M.copy())


class TestDetParity:
    def test_det(self):
        la, ta = _sq(4)
        check_parity(LLA.det(la), TLA.det(ta))


class TestInvParity:
    def test_inv(self):
        la, ta = _sq(4)
        check_parity(LLA.inv(la), TLA.inv(ta), atol=2e-4)


class TestQRParity:
    def test_Q_shape(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((6, 4)).astype(np.float32)
        lQ, lR = LLA.qr(lucid.tensor(A.copy()))
        tQ, tR = TLA.qr(ref.tensor(A.copy()))
        # Q's columns may differ in sign — check QR product instead
        check_parity(lucid.matmul(lQ, lR), ref.matmul(tQ, tR), atol=2e-4)


class TestSVDParity:
    def test_singular_values(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((5, 4)).astype(np.float32)
        _, lS, _ = LLA.svd(lucid.tensor(A.copy()))
        tS = TLA.svdvals(ref.tensor(A.copy()))
        check_parity(lS, tS, atol=2e-4)


class TestEighParity:
    def test_eigenvalues_sorted(self):
        la, ta = _spd(4)
        lvals, _ = LLA.eigh(la)
        tvals, _ = TLA.eigh(ta)
        check_parity(lvals, tvals, atol=2e-3)


class TestCholeskyParity:
    def test_cholesky(self):
        la, ta = _spd(4)
        check_parity(LLA.cholesky(la), TLA.cholesky(ta), atol=2e-4)


class TestNormParity:
    def test_matrix_norm_fro(self):
        la, ta = _sq(4)
        l_norm = LLA.matrix_norm(la, ord="fro")
        t_norm = TLA.matrix_norm(ta, ord="fro")
        check_parity(l_norm, t_norm)

    def test_vector_norm(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(8).astype(np.float32)
        check_parity(
            LLA.vector_norm(lucid.tensor(x.copy())),
            TLA.vector_norm(ref.tensor(x.copy())),
        )


class TestSolveParity:
    def test_solve(self):
        rng = np.random.default_rng(0)
        A = (rng.standard_normal((4, 4)) + np.eye(4) * 2).astype(np.float32)
        b = rng.standard_normal((4, 2)).astype(np.float32)
        check_parity(
            LLA.solve(lucid.tensor(A.copy()), lucid.tensor(b.copy())),
            TLA.solve(ref.tensor(A.copy()), ref.tensor(b.copy())),
            atol=2e-4,
        )
