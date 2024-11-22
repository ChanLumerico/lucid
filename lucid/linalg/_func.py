import numpy as np

from lucid._backend import (
    create_func_op,
    create_bfunc_op,
    create_ufunc_op,
    _FuncOpReturnType,
)
from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar


@create_ufunc_op()
def inv(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.linalg.inv(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return -np.dot(np.dot(result.data.T, result.grad), result.data)

    return result, compute_grad


@create_ufunc_op()
def det(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.linalg.det(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return result.data * np.linalg.inv(self.data).T * result.grad

    return result, compute_grad


@create_bfunc_op()
def solve(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.linalg.solve(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        inv_A = np.linalg.inv(self.data)
        self_grad = -inv_A @ (result.grad @ result.data.T) @ inv_A
        other_grad = inv_A @ result.grad

        return self_grad, other_grad

    return result, compute_grad


@create_ufunc_op()
def cholesky(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.linalg.cholesky(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return 2 * result.data * result.grad

    return result, compute_grad


@create_ufunc_op()
def norm(self: Tensor, ord: int = 2) -> _FuncOpReturnType:
    if not isinstance(ord, int):
        raise NotImplementedError("Only integer p-norms are supported.")

    result = Tensor(np.linalg.norm(self.data, ord=ord))

    def compute_grad() -> _ArrayOrScalar:
        if ord == 2:
            grad = (
                self.data / result.data
                if result.data != 0
                else np.zeros_like(self.data)
            )
        elif ord == 1:
            grad = np.sign(self.data)
        else:
            grad = (
                (np.abs(self.data) ** (ord - 1))
                * np.sign(self.data)
                / (result.data ** (ord - 1))
                if result.data != 0
                else np.zeros_like(self.data)
            )

        return grad * result.grad

    return result, compute_grad


@create_func_op(n_in=1, n_ret=2)
def eig(self: Tensor) -> _FuncOpReturnType:
    eigvals, eigvecs = np.linalg.eig(self.data)
    ndim = self.shape[-2]

    result_eigvals = Tensor(eigvals)
    result_eigvecs = Tensor(eigvecs / np.linalg.norm(eigvecs, axis=-2, keepdims=True))

    def compute_grad_eigvals() -> _ArrayOrScalar:
        grad = np.einsum(
            "...k,...ki,...kj->...ij", result_eigvals.grad, eigvecs, eigvecs
        )
        return grad

    def compute_grad_eigvecs(_eps: float = 1e-12) -> _ArrayOrScalar:
        eigval_diffs = eigvals[..., :, np.newaxis] - eigvals[..., np.newaxis, :]
        eigval_diffs += np.eye(ndim)[..., :, :] * _eps

        inv_eigval_diffs = 1.0 / eigval_diffs
        for index in np.ndindex(inv_eigval_diffs.shape[:-2]):
            np.fill_diagonal(inv_eigval_diffs[index], 0.0)

        outer_prods = np.einsum("...ip,...jq->...pqij", eigvecs, eigvecs)
        S = np.einsum("...kp,...pqij->...pij", inv_eigval_diffs, outer_prods)

        grad = np.einsum("...pk,...pij,...ki->...ij", result_eigvecs.grad, S, eigvecs)
        return grad

    return (
        (result_eigvals, compute_grad_eigvals),
        (result_eigvecs, compute_grad_eigvecs),
    )


@create_func_op(n_in=1, n_ret=2)
def qr(self: Tensor) -> _FuncOpReturnType:
    Q, R = np.linalg.qr(self.data)

    result_q = Tensor(Q)
    result_r = Tensor(R)

    def compute_grad_q() -> _ArrayOrScalar:
        grad_q = result_q.grad
        qt_grad_q = np.einsum("...ik,...kj->...ij", Q.mT, grad_q)
        qt_grad_q_r = np.einsum("...ij,...jk->...ik", qt_grad_q, R)

        return np.einsum("...ij,...jk->...ik", grad_q, R) - np.einsum(
            "...ij,...jk->...ik", Q, qt_grad_q_r
        )

    def compute_grad_r() -> _ArrayOrScalar:
        grad_r = result_r.grad
        return np.einsum("...ij,...jk->...ik", Q, grad_r)

    return (result_q, compute_grad_q), (result_r, compute_grad_r)


@create_func_op(n_in=1, n_ret=3)
def svd(self: Tensor, full_matrices: bool = True) -> _FuncOpReturnType:
    U, S, VT = np.linalg.svd(self.data, full_matrices=full_matrices)

    result_u = Tensor(U)
    result_s = Tensor(S)
    result_vt = Tensor(VT)

    def compute_grad_u() -> _ArrayOrScalar:
        return np.einsum("...ik,...k,...jk->...ij", result_u.grad, S, VT.mT)

    def compute_grad_s() -> _ArrayOrScalar:
        return np.einsum("...ik,...k,...jk->...ij", U, result_s.grad, VT.mT)

    def compute_grad_vt() -> _ArrayOrScalar:
        return np.einsum("...ik,...k,...jk->...ij", U, S, result_vt.grad.mT)

    return (
        (result_u, compute_grad_u),
        (result_s, compute_grad_s),
        (result_vt, compute_grad_vt),
    )


@create_ufunc_op()
def matrix_power(self: Tensor, n: int) -> _FuncOpReturnType:
    result = Tensor(np.linalg.matrix_power(self.data, n))

    def compute_grad() -> _ArrayOrScalar:
        grad = np.zeros_like(self.data)
        if n == 0:
            return grad
        else:
            for k in range(abs(n)):
                left_exp = n - np.sign(n) * k - np.sign(n)
                right_exp = np.sign(n) * k

                left = np.linalg.matrix_power(self.data, left_exp)
                right = np.linalg.matrix_power(self.data, right_exp)

                grad += left @ result.grad @ right
            if n < 0:
                grad = -grad

        return grad

    return result, compute_grad


@create_ufunc_op()
def pinv(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.linalg.pinv(self.data))

    def compute_grad() -> _ArrayOrScalar:
        U, S, Vh = np.linalg.svd(self.data, full_matrices=False)
        S_inv_squared = np.diag(1 / (S**2))

        term_1 = (
            Vh.T
            @ S_inv_squared
            @ U.T
            @ result.grad.T
            @ (np.eye(self.shape[0]) - self.data @ result.data)
        )
        term_2 = (
            (np.eye(self.shape[1]) - result.data @ self.data)
            @ result.grad.T
            @ U
            @ S_inv_squared
            @ Vh
        )
        grad = -result.data.T @ result.grad.T @ result.data.T + term_1 + term_2
        return grad.T

    return result, compute_grad
