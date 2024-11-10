import numpy as np

from lucid._func._common import create_bfunc_op, create_ufunc_op
from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar


@create_ufunc_op()
def inv(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.linalg.inv(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return -np.dot(np.dot(result.data.T, result.grad), result.data)

    return result, compute_grad


@create_ufunc_op()
def det(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.linalg.det(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return result.data * np.linalg.inv(self.data).T * result.grad

    return result, compute_grad


@create_bfunc_op()
def solve(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.linalg.solve(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        inv_A = np.linalg.inv(self.data)
        self_grad = -inv_A @ (result.grad @ result.data.T) @ inv_A
        other_grad = inv_A @ result.grad

        return self_grad, other_grad

    return result, compute_grad


@create_ufunc_op()
def cholesky(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.linalg.cholesky(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return 2 * result.data * result.grad

    return result, compute_grad


@create_ufunc_op()
def norm(self: Tensor, ord: int = 2) -> tuple[Tensor, callable]:
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
