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
        self_grad = -np.linalg.inv(self.data) @ (np.outer(result.grad, other.data).T)
        other_grad = np.linalg.inv(self.data) @ result.grad

        return self_grad, other_grad

    return result, compute_grad


@create_ufunc_op()
def cholesky(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.linalg.cholesky(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return 2 * result.data * result.grad

    return result, compute_grad
