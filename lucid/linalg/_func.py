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
