import numpy as np

from lucid.tensor import Tensor, _ArrayOrScalar
from lucid._func._common import create_ufunc_op


@create_ufunc_op
def pow(self: Tensor, exp: int | float) -> Tensor:
    result = Tensor(self.data**exp, requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return exp * self.data ** (exp - 1)

    return result, compute_grad


@create_ufunc_op
def neg(self: Tensor) -> Tensor:
    """Negation (Unary minus)"""
    result = Tensor(-self.data, requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return -1

    return result, compute_grad


@create_ufunc_op
def exp(self: Tensor) -> Tensor:
    """Exponential function"""
    result = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return result.data

    return result, compute_grad


@create_ufunc_op
def log(self: Tensor) -> Tensor:
    """Natural logarithm"""
    result = Tensor(np.log(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return 1 / self.data

    return result, compute_grad


@create_ufunc_op
def sqrt(self: Tensor) -> Tensor:
    """Square root"""
    result = Tensor(np.sqrt(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return 0.5 / result.data

    return result, compute_grad


@create_ufunc_op
def sin(self: Tensor) -> Tensor:
    """Sine function"""
    result = Tensor(np.sin(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return np.cos(self.data)

    return result, compute_grad


@create_ufunc_op
def cos(self: Tensor) -> Tensor:
    """Cosine function"""
    result = Tensor(np.cos(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return -np.sin(self.data)

    return result, compute_grad


@create_ufunc_op
def tan(self: Tensor) -> Tensor:
    """Tangent function"""
    result = Tensor(np.tan(self.data), requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return 1 / (np.cos(self.data) ** 2)

    return result, compute_grad
