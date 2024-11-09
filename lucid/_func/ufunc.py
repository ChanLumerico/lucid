import numpy as np

from lucid.tensor import Tensor, _ArrayOrScalar
from lucid._func._common import create_ufunc_op


@create_ufunc_op
def pow(self: Tensor, exp: int | float) -> tuple[Tensor, callable]:
    result = Tensor(self.data**exp, requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return exp * self.data ** (exp - 1)

    return result, compute_grad


@create_ufunc_op
def neg(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(-self.data, requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return -1

    return result, compute_grad


@create_ufunc_op
def exp(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return result.data

    return result, compute_grad


@create_ufunc_op
def log(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.log(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return 1 / self.data

    return result, compute_grad


@create_ufunc_op
def sqrt(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.sqrt(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return 0.5 / result.data

    return result, compute_grad


@create_ufunc_op
def sin(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.sin(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return np.cos(self.data)

    return result, compute_grad


@create_ufunc_op
def cos(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.cos(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return -np.sin(self.data)

    return result, compute_grad


@create_ufunc_op
def tan(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.tan(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return 1 / (np.cos(self.data) ** 2)

    return result, compute_grad


@create_ufunc_op
def clip(
    self: Tensor,
    min_value: float,
    max_value: float,
) -> tuple[Tensor, callable]:
    result = Tensor(
        np.clip(self.data, min_value, max_value),
        requires_grad=self.requires_grad,
    )

    def compute_grad() -> _ArrayOrScalar:
        grad = np.ones_like(self.data)
        grad[self.data < min_value] = 0
        grad[self.data > max_value] = 0
        return grad

    return result, compute_grad
