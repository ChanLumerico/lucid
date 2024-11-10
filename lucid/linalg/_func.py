import numpy as np

from lucid._func._common import create_bfunc_op, create_ufunc_op
from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar


@create_ufunc_op()
def inv(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.linalg.inv(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return -np.dot(result.data.T, result.data)

    return result, compute_grad


@create_ufunc_op()
def det(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.linalg.det(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return np.linalg.inv(self.data).T

    return result, compute_grad
