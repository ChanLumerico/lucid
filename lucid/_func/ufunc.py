from typing import Optional
import numpy as np

from lucid._tensor import Tensor
from lucid._func._common import create_ufunc_op
from lucid.types import _ArrayOrScalar


@create_ufunc_op()
def _pow(self: Tensor, exp: int | float) -> tuple[Tensor, callable]:
    result = Tensor(self.data**exp)

    def compute_grad() -> _ArrayOrScalar:
        return (exp * self.data ** (exp - 1)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def _neg(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(-self.data)

    def compute_grad() -> _ArrayOrScalar:
        return -result.grad

    return result, compute_grad


@create_ufunc_op()
def exp(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.exp(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return result.data * result.grad

    return result, compute_grad


@create_ufunc_op()
def log(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.log(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return (1 / self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sqrt(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.sqrt(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return (0.5 / result.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sin(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.sin(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return np.cos(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def cos(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.cos(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return -np.sin(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def tan(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.tan(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return (1 / (np.cos(self.data) ** 2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arcsin(self: Tensor) -> Tensor:
    result = Tensor(np.arcsin(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return (1 / np.sqrt(1 - self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arccos(self: Tensor) -> Tensor:
    result = Tensor(np.arccos(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return (-1 / np.sqrt(1 - self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arctan(self: Tensor) -> Tensor:
    result = Tensor(np.arctan(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return (1 / (1 + self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sinh(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.sinh(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return np.cosh(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def cosh(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.cosh(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return np.sinh(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def tanh(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.tanh(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return (1 - np.tanh(self.data) ** 2) * result.grad

    return result, compute_grad


@create_ufunc_op()
def clip(self: Tensor, min_value: float, max_value: float) -> tuple[Tensor, callable]:
    result = Tensor(np.clip(self.data, min_value, max_value))

    def compute_grad() -> _ArrayOrScalar:
        grad = np.ones_like(self.data)
        grad[self.data < min_value] = 0
        grad[self.data > max_value] = 0
        return grad * result.grad

    return result, compute_grad


@create_ufunc_op()
def abs(self: Tensor) -> Tensor:
    result = Tensor(np.abs(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return np.where(self.data >= 0, 1, -1) * result.grad

    return result, compute_grad


@create_ufunc_op(has_gradient=False)
def sign(self: Tensor) -> Tensor:
    result = Tensor(np.sign(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return 0

    return result, compute_grad


@create_ufunc_op()
def reciprocal(self: Tensor) -> Tensor:
    result = Tensor(1 / self.data)

    def compute_grad() -> _ArrayOrScalar:
        return (-1 / (self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def square(self: Tensor) -> Tensor:
    result = Tensor(np.square(self.data))

    def compute_grad() -> _ArrayOrScalar:
        return 2 * self.data * result.grad

    return result, compute_grad


@create_ufunc_op()
def cube(self: Tensor) -> Tensor:
    result = Tensor(self.data**3)

    def compute_grad() -> _ArrayOrScalar:
        return 3 * self.data**2 * result.grad

    return result, compute_grad


@property
@create_ufunc_op()
def _T(self: Tensor) -> Tensor:
    result = Tensor(self.data.T)

    def compute_grad() -> _ArrayOrScalar:
        return result.grad.T

    return result, compute_grad


@create_ufunc_op()
def transpose(
    self: Tensor, axes: Optional[list[int]] = None
) -> tuple[Tensor, callable]:
    if axes is None:
        axes = list(reversed(range(self.ndim)))
    result = Tensor(np.transpose(self.data, axes))

    def compute_grad() -> _ArrayOrScalar:
        return np.transpose(result.grad, np.argsort(axes))

    return result, compute_grad


@create_ufunc_op()
def sum(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> tuple[Tensor, callable]:
    result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _ArrayOrScalar:
        grad_shape = list(result.grad.shape)
        if axis is not None and not keepdims:
            axis_tuple = axis if isinstance(axis, tuple) else (axis,)
            for ax in axis_tuple:
                grad_shape.insert(ax, 1)

        grad = np.reshape(result.grad, grad_shape)
        if grad.shape != self.data.shape:
            grad = np.broadcast_to(grad, self.data.shape)

        return grad

    return result, compute_grad


@create_ufunc_op()
def trace(self: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.trace(self.data))

    def compute_grad() -> _ArrayOrScalar:
        grad = np.zeros_like(self.data)
        np.fill_diagonal(grad, 1)
        return grad * result.grad

    return result, compute_grad
