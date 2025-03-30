from types import ModuleType
from functools import partial
import numpy as np

from lucid.types import _NumPyArray, _MLXArray
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
        fallback_ = self.axis is None or (
            isinstance(self.axis, (tuple, list)) and a.ndim > 1
        )
        if fallback_:
            result_data = np.linalg.norm(
                a.data, ord=self.ord, axis=self.axis, keepdims=self.keepdims
            )
            self.result = Tensor(result_data, device="gpu")
        else:
            result_data = mx.linalg.norm(
                a.data, ord=self.ord, axis=self.axis, keepdims=self.keepdims
            )
            self.result = Tensor(result_data)

        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
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

        return grad * grad_output


@fallback
class eig(operation):
    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def _unified(self, a: Tensor) -> tuple[_NumPyArray, _NumPyArray]:
        self.ndim = a.shape[-2]
        eigvals, eigvecs = np.linalg.eig(a.data)
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=-2, keepdims=True)

        return eigvals, eigvecs

    @func_op(n_in=1, n_ret=2)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self._eigvals, self._eigvecs = self._unified(a)
        self.result = (Tensor(self._eigvals), Tensor(self._eigvecs))

        return (
            (self.result[0], self.compute_grad_eigvals),
            (self.result[1], self.compute_grad_eigvecs),
        )

    @func_op(n_in=1, n_ret=2, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self._eigvals, self._eigvecs = self._unified(a)
        self.result = (
            Tensor(self._eigvals, device="gpu"),
            Tensor(self._eigvecs, device="gpu"),
        )

        return (
            (self.result[0], partial(self.compute_grad_eigvals, _fallback=True)),
            (self.result[1], partial(self.compute_grad_eigvecs, _fallback=True)),
        )

    def compute_grad_eigvals(self, _fallback: bool = False) -> _GradFuncType:
        eigvals = self.result[0]
        grad = np.einsum(
            "...k,...ki,...kj->...ij",
            np.array(eigvals.grad),
            self._eigvecs,
            self._eigvecs,
        )
        return grad if not _fallback else mx.array(grad)

    def compute_grad_eigvecs(self, _fallback: bool = False) -> _GradFuncType:
        eigvals, eigvecs = self._eigvals, self._eigvecs
        eigvecs_t = self.result[1]

        eigval_diffs = eigvals[..., :, np.newaxis] - eigvals[..., np.newaxis, :]
        eigval_diffs += np.eye(self.ndim)[..., :, :] * self.eps

        inv_eigval_diffs = 1.0 / eigval_diffs
        for index in np.ndindex(inv_eigval_diffs.shape[:-2]):
            np.fill_diagonal(inv_eigval_diffs[index], 0.0)

        outer_prods = np.einsum("...ip,...jq->...pqij", eigvecs, eigvecs)
        S = np.einsum("...kp,...pqij->...pij", inv_eigval_diffs, outer_prods)

        grad = np.einsum(
            "...pk,...pij,...ki->...ij", np.array(eigvecs_t.grad), S, eigvecs
        )
        return grad if not _fallback else mx.array(grad)


class qr(operation):
    def __init__(self) -> None:
        super().__init__()

    @func_op(n_in=1, n_ret=2)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.Q, self.R = np.linalg.qr(a.data)
        self.result = (Tensor(self.Q), Tensor(self.R))

        return (
            (self.result[0], partial(self.compute_grad_q, lib_=np)),
            (self.result[1], partial(self.compute_grad_r, lib_=np)),
        )

    @func_op(n_in=1, n_ret=2, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.Q, self.R = mx.linalg.qr(a.data)
        self.result = (Tensor(self.Q), Tensor(self.R))

        return (
            (self.result[0], partial(self.compute_grad_q, lib_=mx)),
            (self.result[1], partial(self.compute_grad_r, lib_=mx)),
        )

    def compute_grad_q(self, lib_: ModuleType) -> _GradFuncType:
        grad_q = self.result[0].grad
        qt_grad_q = lib_.einsum("...ik,...kj->...ij", self.Q.swapaxes(-1, -2), grad_q)
        qt_grad_q_r = lib_.einsum("...ij,...jk->...ik", qt_grad_q, self.R)

        return lib_.einsum("...ij,...jk->...ik", grad_q, self.R) - lib_.einsum(
            "...ij,...jk->...ik", self.Q, qt_grad_q_r
        )

    def compute_grad_r(self, lib_: ModuleType) -> _GradFuncType:
        grad_r = self.result[1].grad
        return lib_.einsum("...ij,...jk->...ik", self.Q, grad_r)


@fallback
class svd(operation):
    def __init__(self, full_matrices: bool) -> None:
        super().__init__()
        self.full_matrices = full_matrices

    @func_op(n_in=1, n_ret=3)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.U, self.S, self.VT = np.linalg.svd(a.data, self.full_matrices)
        self.result = (Tensor(self.U), Tensor(self.S), Tensor(self.VT))

        return (
            (self.result[0], partial(self.compute_grad_u, lib_=np)),
            (self.result[1], partial(self.compute_grad_s, lib_=np)),
            (self.result[2], partial(self.compute_grad_vt, lib_=np)),
        )

    @func_op(n_in=1, n_ret=3, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        if self.full_matrices:
            U, S, VT = np.linalg.svd(np.array(a.data), full_matrices=True)
            self.U, self.S, self.VT = mx.array(U), mx.array(S), mx.array(VT)
        else:
            self.U, self.S, self.VT = mx.linalg.svd(a.data)

        self.result = (Tensor(self.U), Tensor(self.S), Tensor(self.VT))

        return (
            (self.result[0], partial(self.compute_grad_u, lib_=mx)),
            (self.result[1], partial(self.compute_grad_s, lib_=mx)),
            (self.result[2], partial(self.compute_grad_vt, lib_=mx)),
        )

    def compute_grad_u(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.einsum(
            "...ik,...k,...jk->...ij",
            self.result[0].grad,
            self.S,
            self.VT.swapaxes(-1, -2),
        )

    def compute_grad_s(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.einsum(
            "...ik,...k,...jk->...ij",
            self.U,
            self.result[1].grad,
            self.VT.swapaxes(-1, -2),
        )

    def compute_grad_vt(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.einsum(
            "...ik,...k,...jk->...ij",
            self.U,
            self.S,
            self.result[2].grad.swapaxes(-1, -2),
        )


class matrix_power(operation):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def _gpu_matrix_pow(self, arr: _MLXArray, n: int) -> _MLXArray:
        if n == 0:
            return mx.eye(arr.shape[0], dtype=arr.dtype)
        elif n < 0:
            arr = mx.linalg.inv(arr)
            n = -n

        result = arr
        for _ in range(n - 1):
            result = result @ arr
        return result

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.linalg.matrix_power(a.data, n=self.n))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(self._gpu_matrix_pow(a.data, n=self.n))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        grad = lib_.zeros_like(a.data)
        if self.n == 0:
            return grad
        else:
            for k in range(abs(self.n)):
                left_exp = self.n - lib_.sign(self.n) * k - lib_.sign(self.n)
                right_exp = lib_.sign(self.n) * k

                left = (
                    np.linalg.matrix_power(a.data, left_exp)
                    if lib_ is np
                    else self._gpu_matrix_pow(a.data, left_exp.item())
                )
                right = (
                    np.linalg.matrix_power(a.data, right_exp)
                    if lib_ is np
                    else self._gpu_matrix_pow(a.data, right_exp.item())
                )

                grad += left @ self.result.grad @ right
            if self.n < 0:
                grad = -grad

        return grad


class pinv(operation):
    def __init__(self, rcond: float) -> None:
        super().__init__()
        self.rcond = rcond

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.linalg.pinv(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        U, S, VT = mx.linalg.svd(a.data)
        S_inv = mx.where(S > self.rcond, 1.0 / S, 0.0)
        S_inv_mat = mx.diag(S_inv)

        self.result = Tensor(VT.T @ S_inv_mat @ U.T)
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        U, S, Vh = (
            np.linalg.svd(a.data, full_matrices=False)
            if lib_ is np
            else mx.linalg.svd(a.data)
        )
        S_inv_squared = lib_.diag(1 / (S**2))
        term_1 = (
            Vh.T
            @ S_inv_squared
            @ U.T
            @ self.result.grad.T
            @ (lib_.eye(a.shape[0]) - a.data @ self.result.data)
        )
        term_2 = (
            (lib_.eye(a.shape[1]) - self.result.data @ a.data)
            @ self.result.grad.T
            @ U
            @ S_inv_squared
            @ Vh
        )
        grad = (
            -self.result.data.T @ self.result.grad.T @ self.result.data.T
            + term_1
            + term_2
        ).T
        return grad
