from types import ModuleType
from functools import partial
import numpy as np

from lucid.types import _ArrayOrScalar, _NumPyArray
from lucid._tensor import Tensor

from lucid._backend.core import (
    operation,
    fallback,
    func_op,
    binary_func_op,
    unary_func_op,
    _GradFuncType,
    _FuncOpReturnType,
)
from lucid._backend.metal import mx


class inv(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.linalg.inv(a.data))
        return self.result, self.compute_grad_cpu

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.linalg.inv(a.data))
        return self.result, self.compute_grad_gpu

    def compute_grad_cpu(self) -> _GradFuncType:
        return -np.dot(np.dot(self.result.data.T, self.result.grad), self.result.data)

    def compute_grad_gpu(self) -> _GradFuncType:
        return -mx.matmul(
            mx.matmul(self.result.data.T, self.result.grad), self.result.data
        )


class det(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.linalg.det(a.data))
        return self.result, partial(self.compute_grad_cpu, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        _, _, U = mx.linalg.lu(a.data)
        diag = mx.diagonal(U)

        self.result = Tensor(mx.prod(diag))
        return self.result, partial(self.compute_grad_gpu, a=a)

    def compute_grad_cpu(self, a: Tensor) -> _GradFuncType:
        grad = self.result.grad
        invA_T = np.transpose(np.linalg.inv(a.data))
        return grad * invA_T

    def compute_grad_gpu(self, a: Tensor) -> _GradFuncType:
        grad = self.result.grad
        invA_T = mx.transpose(mx.linalg.inv(a.data))
        return grad * invA_T


class solve(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.linalg.solve(a.data, b.data))
        return self.result, partial(self.compute_grad_cpu, a=a)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.linalg.solve(a.data, b.data))
        return self.result, partial(self.compute_grad_gpu, a=a)

    def compute_grad_cpu(self, a: Tensor) -> _GradFuncType:
        grad = self.result.grad
        x = self.result.data
        inv_a = np.linalg.inv(a.data)

        a_grad = -inv_a @ (grad @ x.T) @ inv_a
        b_grad = inv_a @ grad

        return a_grad, b_grad

    def compute_grad_gpu(self, a: Tensor) -> _GradFuncType:
        grad = self.result.grad
        x = self.result.data
        inv_a = mx.linalg.inv(a.data)

        a_grad = -mx.matmul(inv_a, mx.matmul(mx.matmul(grad, x.T), inv_a))
        b_grad = mx.matmul(inv_a, grad)

        return a_grad, b_grad


class cholesky(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.linalg.cholesky(a.data))
        return self.result, partial(self.compute_grad, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.linalg.cholesky(a.data))
        return self.result, partial(self.compute_grad, lib_=mx)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        L = self.result.data
        grad_L = self.result.grad

        L_inv = lib_.linalg.inv(L)
        inner = L.T @ grad_L
        sym = 0.5 * (inner + inner.T)

        return L_inv.T @ (sym @ L_inv)


@fallback
class norm(operation):
    def __init__(
        self,
        ord: int = 2,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> None:
        super().__init__()
        self.ord = ord
        self.axis = axis
        self.keepdims = keepdims

        if not isinstance(self.ord, int):
            raise NotImplementedError("Only integer p-norms are supported.")

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        result_data = np.linalg.norm(
            a.data, ord=self.ord, axis=self.axis, keepdims=self.keepdims
        )
        self.result = Tensor(result_data)
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        global_matrix_norm = self.axis is None or (
            isinstance(self.axis, (tuple, list)) and a.ndim > 1
        )

        if global_matrix_norm:
            result_data = np.linalg.norm(
                a.data, ord=self.ord, axis=self.axis, keepdims=self.keepdims
            )
            self.result = Tensor(result_data, device="gpu")
            return self.result, partial(self.compute_grad, a=a, lib_=np, _fallback=True)

        result_data = mx.linalg.norm(
            a.data, ord=self.ord, axis=self.axis, keepdims=self.keepdims
        )
        self.result = Tensor(result_data)
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(
        self, a: Tensor, lib_: ModuleType, _fallback: bool = False
    ) -> _GradFuncType:
        if _fallback:
            x = np.array(a.data)
            r = np.array(self.result.data)
            grad_output = np.array(self.result.grad)
        else:
            x = a.data
            r = self.result.data
            grad_output = self.result.grad

        ord = self.ord
        axis = self.axis
        keepdims = self.keepdims

        if ord == 2:
            denom = r
            if not keepdims and axis is not None:
                denom = lib_.expand_dims(r, axis=axis)
            grad = lib_.where(lib_.all(r != 0), x / denom, lib_.zeros_like(x))

        elif ord == 1:
            grad = lib_.sign(x)

        else:
            denom = r
            if not keepdims and axis is not None:
                denom = lib_.expand_dims(r, axis=axis)
            grad = lib_.where(
                lib_.all(r != 0),
                (lib_.abs(x) ** (ord - 1)) * lib_.sign(x) / (denom ** (ord - 1)),
                lib_.zeros_like(x),
            )

        if axis is not None and not keepdims:
            grad_output = lib_.expand_dims(grad_output, axis=axis)

        grad_final = grad * grad_output
        return grad_final if not _fallback else mx.array(grad_final)


@fallback
class eig(operation):
    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def _unified(self, a: Tensor) -> tuple[_NumPyArray, _NumPyArray]:
        eigvals, eigvecs = np.linalg.eig(a.data)
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=-2, keepdims=True)
        return eigvals, eigvecs

    @func_op(n_in=1, n_ret=2)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        eigvals, eigvecs = self._unified(a)
        self.result = (Tensor(eigvals), Tensor(eigvecs))

        return (
            (self.result[0], self.compute_grad_eigvals),
            (self.result[1], partial(self.compute_grad_eigvecs, a=a)),
        )

    @func_op(n_in=1, n_ret=2, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        eigvals, eigvecs = self._unified(a)
        self.result = (Tensor(eigvals, device="gpu"), Tensor(eigvecs, device="gpu"))

        return (
            (self.result[0], partial(self.compute_grad_eigvals, _fallback=True)),
            (self.result[1], partial(self.compute_grad_eigvecs, _fallback=True)),
        )

    def compute_grad_eigvals(self, _fallback: bool = False) -> _GradFuncType:
        eigvals, eigvecs = self.result
        if _fallback:
            eigvals, eigvecs = np.array(eigvals), np.array(eigvecs)

        grad = np.einsum("...k,...ki,...kj->...ij", eigvals.grad, eigvecs, eigvecs)
        return grad if not _fallback else mx.array(grad)

    def compute_grad_eigvecs(self, _fallback: bool = False) -> _GradFuncType:
        eigvals, eigvecs = self.result
        if _fallback:
            eigvals, eigvecs = np.array(eigvals), np.array(eigvecs)

        eigval_diffs = eigvals[..., :, np.newaxis] - eigvals[..., np.newaxis, :]
        ndim = eigvals.shape[-1]
        eigval_diffs += np.eye(ndim)[..., :, :] * self.eps

        inv_eigval_diffs = 1.0 / eigval_diffs
        for index in np.ndindex(inv_eigval_diffs.shape[:-2]):
            np.fill_diagonal(inv_eigval_diffs[index], 0.0)

        outer_prods = np.einsum("...ip,...jq->...pqij", eigvecs, eigvecs)
        S = np.einsum("...kp,...pqij->...pij", inv_eigval_diffs, outer_prods)

        grad = np.einsum("...pk,...pij,...ki->...ij", eigvecs.grad, S, eigvecs)
        return grad if not _fallback else mx.array(grad)


@func_op(n_in=1, n_ret=2)
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


@func_op(n_in=1, n_ret=3)
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


@unary_func_op()
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


@unary_func_op()
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
