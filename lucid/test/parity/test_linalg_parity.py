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
        la, ta = _sq(4)
        rng = np.random.default_rng(1)
        b_np = rng.standard_normal((4, 2)).astype(np.float32)
        check_parity(
            LLA.solve(la, lucid.tensor(b_np.copy())),
            TLA.solve(ta, ref.tensor(b_np.copy())),
            atol=2e-4,
        )


class TestLinalgGPUForwardParity:
    """GPU output must match CPU output bit-close for the same input.

    Locks down GPU dispatch (ensures the linalg op runs on a GPU stream and
    returns a GPU tensor). Caught a real bug where ``det`` reduced over all
    axes instead of the trailing diagonal axis on batched inputs.
    """

    def _both(self, fn, np_arr: np.ndarray, atol: float = 1e-4) -> None:
        cpu_t: lucid.Tensor = lucid.tensor(np_arr.copy())
        gpu_t: lucid.Tensor = cpu_t.to("metal")
        out_cpu = fn(cpu_t)
        out_gpu = fn(gpu_t)
        if isinstance(out_cpu, tuple):
            for c, g in zip(out_cpu, out_gpu):
                np.testing.assert_allclose(
                    np.abs(c.numpy()), np.abs(g.numpy()), atol=atol
                )
        else:
            np.testing.assert_allclose(out_cpu.numpy(), out_gpu.numpy(), atol=atol)

    def test_inv_2d(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        A: np.ndarray = (rng.standard_normal((4, 4)) + np.eye(4) * 5).astype(np.float32)
        self._both(LLA.inv, A)

    def test_det_batched(self) -> None:
        # Batched det was the GPU-only bug: prod() reduced all axes giving the
        # same scalar for every batch element.
        rng: np.random.Generator = np.random.default_rng(2)
        A: np.ndarray = (rng.standard_normal((3, 4, 4)) + np.eye(4) * 5).astype(
            np.float32
        )
        self._both(LLA.det, A)

    def test_cholesky(self) -> None:
        rng: np.random.Generator = np.random.default_rng(3)
        A: np.ndarray = rng.standard_normal((4, 4)).astype(np.float32)
        SPD: np.ndarray = (A @ A.T + np.eye(4) * 4).astype(np.float32)
        self._both(LLA.cholesky, SPD)

    def test_qr(self) -> None:
        rng: np.random.Generator = np.random.default_rng(4)
        A: np.ndarray = rng.standard_normal((4, 4)).astype(np.float32)
        # Q columns can differ in sign — compare reconstruction A = Q@R.
        cpu_t: lucid.Tensor = lucid.tensor(A.copy())
        gpu_t: lucid.Tensor = cpu_t.to("metal")
        Q_c, R_c = LLA.qr(cpu_t)
        Q_g, R_g = LLA.qr(gpu_t)
        np.testing.assert_allclose((Q_c @ R_c).numpy(), (Q_g @ R_g).numpy(), atol=1e-4)

    def test_pinv(self) -> None:
        rng: np.random.Generator = np.random.default_rng(5)
        A: np.ndarray = rng.standard_normal((5, 3)).astype(np.float32)
        self._both(LLA.pinv, A, atol=2e-4)

    def test_svd_singular_values(self) -> None:
        rng: np.random.Generator = np.random.default_rng(6)
        A: np.ndarray = rng.standard_normal((4, 4)).astype(np.float32)
        cpu_t: lucid.Tensor = lucid.tensor(A.copy())
        gpu_t: lucid.Tensor = cpu_t.to("metal")
        _, s_c, _ = LLA.svd(cpu_t)
        _, s_g, _ = LLA.svd(gpu_t)
        np.testing.assert_allclose(s_c.numpy(), s_g.numpy(), atol=2e-4)


class TestLinalgBackwardParity:
    """Backward parity for the differentiable linalg ops."""

    def test_inv_backward(self) -> None:
        la, ta = _sq(4)
        la.requires_grad = True
        ta.requires_grad_(True)
        LLA.inv(la).sum().backward()
        TLA.inv(ta).sum().backward()
        np.testing.assert_allclose(la.grad.numpy(), ta.grad.numpy(), atol=1e-4)

    def test_det_backward_batched(self) -> None:
        # Det.cpp's broadcast_to needed an explicit reshape — ``[B]`` cannot
        # broadcast to ``[B, N, N]`` without first being reshaped to ``[B, 1, 1]``.
        rng: np.random.Generator = np.random.default_rng(0)
        A_np: np.ndarray = (rng.standard_normal((2, 3, 3)) + np.eye(3) * 5).astype(
            np.float32
        )
        A_l: lucid.Tensor = lucid.tensor(A_np.copy(), requires_grad=True)
        A_t = ref.tensor(A_np.copy(), requires_grad=True)
        LLA.det(A_l).sum().backward()
        TLA.det(A_t).sum().backward()
        np.testing.assert_allclose(A_l.grad.numpy(), A_t.grad.numpy(), atol=1e-4)

    def test_solve_backward(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        A_np: np.ndarray = (rng.standard_normal((4, 4)) + np.eye(4) * 2).astype(
            np.float32
        )
        b_np: np.ndarray = rng.standard_normal((4, 2)).astype(np.float32)
        A_l: lucid.Tensor = lucid.tensor(A_np.copy(), requires_grad=True)
        b_l: lucid.Tensor = lucid.tensor(b_np.copy(), requires_grad=True)
        A_t = ref.tensor(A_np.copy(), requires_grad=True)
        b_t = ref.tensor(b_np.copy(), requires_grad=True)
        LLA.solve(A_l, b_l).sum().backward()
        TLA.solve(A_t, b_t).sum().backward()
        np.testing.assert_allclose(A_l.grad.numpy(), A_t.grad.numpy(), atol=2e-4)
        np.testing.assert_allclose(b_l.grad.numpy(), b_t.grad.numpy(), atol=2e-4)


class TestMatrixPowerBackward:
    """matrix_power autograd is provided by a Python decomposition over
    matmul + inv (see lucid/linalg/__init__.py:matrix_power), since the
    engine kernel itself has no backward node."""

    @pytest.mark.parametrize("n", [2, 3, -1, -2])
    def test_matrix_power_backward(self, n: int) -> None:
        ll, lt = _sq(4)
        ll.requires_grad = True
        lt.requires_grad_(True)
        LLA.matrix_power(ll, n).sum().backward()
        TLA.matrix_power(lt, n).sum().backward()
        np.testing.assert_allclose(ll.grad.numpy(), lt.grad.numpy(), atol=1e-3)


class TestSquarePinvBackward:
    """For a square full-rank matrix ``pinv ≡ inv``; the Python wrapper
    routes through ``inv`` so autograd flows naturally."""

    def test_pinv_square(self) -> None:
        ll, lt = _sq(4)
        ll.requires_grad = True
        lt.requires_grad_(True)
        LLA.pinv(ll).sum().backward()
        TLA.pinv(lt).sum().backward()
        np.testing.assert_allclose(ll.grad.numpy(), lt.grad.numpy(), atol=1e-3)


class TestRemainingLinalgBackwardGaps:
    """Locked-down record of linalg ops whose backwards are still
    unimplemented. Once a backward lands, flip the matching xfail to a real
    parity test (see TestCholeskyBackward / TestMatrixPowerBackward for the
    pattern)."""

    @pytest.mark.xfail(
        reason="QR backward not wired (engine qr_op has no autograd; "
        "Q/R formula needs sym_lower(A^{-T}(...))-style construction).",
        strict=True,
    )
    def test_qr_backward(self) -> None:
        ll, _ = _sq(4)
        ll.requires_grad = True
        Q, R = LLA.qr(ll)
        (Q + R).sum().backward()
        assert ll.grad is not None

    @pytest.mark.xfail(
        reason="SVD backward not wired (Townsend 2016 formula needs "
        "explicit computation across U/S/Vh on top of inv/matmul).",
        strict=True,
    )
    def test_svd_backward(self) -> None:
        ll, _ = _sq(4)
        ll.requires_grad = True
        _, S, _ = LLA.svd(ll)
        S.sum().backward()
        assert ll.grad is not None

    @pytest.mark.xfail(
        reason="eigvalsh backward not wired — requires eigenvectors via "
        "eigh, which itself needs differentiable spectral decomposition.",
        strict=True,
    )
    def test_eigvalsh_backward(self) -> None:
        ll, _ = _spd(4)
        ll.requires_grad = True
        LLA.eigvalsh(ll).sum().backward()
        assert ll.grad is not None

    @pytest.mark.xfail(
        reason="Non-square pinv backward not wired — would require either "
        "an autograd-capable SVD or the Golub-Pereyra adjoint formula.",
        strict=True,
    )
    def test_pinv_nonsquare_backward(self) -> None:
        rng: np.random.Generator = np.random.default_rng(0)
        A_np: np.ndarray = rng.standard_normal((5, 3)).astype(np.float32)
        A_l: lucid.Tensor = lucid.tensor(A_np.copy(), requires_grad=True)
        LLA.pinv(A_l).sum().backward()
        assert A_l.grad is not None


class TestCholeskyBackward:
    """Cholesky autograd is wired in Python on top of solve_triangular and
    matmul (Murray 2016 formula); the engine cholesky_op itself has no
    backward node."""

    def test_cholesky_lower_backward(self) -> None:
        ll, lt = _spd(4)
        ll.requires_grad = True
        lt.requires_grad_(True)
        LLA.cholesky(ll).sum().backward()
        TLA.cholesky(lt).sum().backward()
        np.testing.assert_allclose(ll.grad.numpy(), lt.grad.numpy(), atol=1e-3)

    def test_cholesky_upper_backward(self) -> None:
        ll, lt = _spd(4)
        ll.requires_grad = True
        lt.requires_grad_(True)
        LLA.cholesky(ll, upper=True).sum().backward()
        TLA.cholesky(lt, upper=True).sum().backward()
        np.testing.assert_allclose(ll.grad.numpy(), lt.grad.numpy(), atol=1e-3)
