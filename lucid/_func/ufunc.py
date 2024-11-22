from typing import Optional
import numpy as np

from lucid._tensor import Tensor
from lucid._backend import create_ufunc_op, _FuncOpReturnType
from lucid.types import _NumPyArray, _Scalar


@create_ufunc_op()
def _pow(self: Tensor, exp: _Scalar) -> _FuncOpReturnType:
    result = Tensor(self.data**exp)

    def compute_grad() -> _NumPyArray:
        return (exp * self.data ** (exp - 1)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def _neg(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(-self.data)

    def compute_grad() -> _NumPyArray:
        return -result.grad

    return result, compute_grad


@create_ufunc_op()
def exp(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.exp(self.data))

    def compute_grad() -> _NumPyArray:
        return result.data * result.grad

    return result, compute_grad


@create_ufunc_op()
def log(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.log(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sqrt(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.sqrt(self.data))

    def compute_grad() -> _NumPyArray:
        return (0.5 / result.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sin(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.sin(self.data))

    def compute_grad() -> _NumPyArray:
        return np.cos(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def cos(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.cos(self.data))

    def compute_grad() -> _NumPyArray:
        return -np.sin(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def tan(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.tan(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / (np.cos(self.data) ** 2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arcsin(self: Tensor) -> Tensor:
    result = Tensor(np.arcsin(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / np.sqrt(1 - self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arccos(self: Tensor) -> Tensor:
    result = Tensor(np.arccos(self.data))

    def compute_grad() -> _NumPyArray:
        return (-1 / np.sqrt(1 - self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arctan(self: Tensor) -> Tensor:
    result = Tensor(np.arctan(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / (1 + self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sinh(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.sinh(self.data))

    def compute_grad() -> _NumPyArray:
        return np.cosh(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def cosh(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.cosh(self.data))

    def compute_grad() -> _NumPyArray:
        return np.sinh(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def tanh(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.tanh(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 - np.tanh(self.data) ** 2) * result.grad

    return result, compute_grad


@create_ufunc_op()
def clip(self: Tensor, min_value: float, max_value: float) -> _FuncOpReturnType:
    result = Tensor(np.clip(self.data, min_value, max_value))

    def compute_grad() -> _NumPyArray:
        grad = np.ones_like(self.data)
        grad[self.data < min_value] = 0
        grad[self.data > max_value] = 0
        return grad * result.grad

    return result, compute_grad


@create_ufunc_op()
def abs(self: Tensor) -> Tensor:
    result = Tensor(np.abs(self.data))

    def compute_grad() -> _NumPyArray:
        return np.where(self.data >= 0, 1, -1) * result.grad

    return result, compute_grad


@create_ufunc_op(has_gradient=False)
def sign(self: Tensor) -> Tensor:
    result = Tensor(np.sign(self.data))

    def compute_grad() -> _NumPyArray:
        return np.array(0.0)

    return result, compute_grad


@create_ufunc_op()
def reciprocal(self: Tensor) -> Tensor:
    result = Tensor(1 / self.data)

    def compute_grad() -> _NumPyArray:
        return (-1 / (self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def square(self: Tensor) -> Tensor:
    result = Tensor(np.square(self.data))

    def compute_grad() -> _NumPyArray:
        return 2 * self.data * result.grad

    return result, compute_grad


@create_ufunc_op()
def cube(self: Tensor) -> Tensor:
    result = Tensor(self.data**3)

    def compute_grad() -> _NumPyArray:
        return 3 * self.data**2 * result.grad

    return result, compute_grad


@property
@create_ufunc_op()
def _T(self: Tensor) -> Tensor:
    result = Tensor(self.data.T)

    def compute_grad() -> _NumPyArray:
        return result.grad.T

    return result, compute_grad


@property
@create_ufunc_op()
def _mT(self: Tensor) -> Tensor:
    result = Tensor(self.data.mT)

    def compute_grad() -> _NumPyArray:
        return result.grad.mT

    return result, compute_grad


@create_ufunc_op()
def transpose(self: Tensor, axes: Optional[list[int]] = None) -> _FuncOpReturnType:
    if axes is None:
        axes = list(reversed(range(self.ndim)))
    result = Tensor(np.transpose(self.data, axes))

    def compute_grad() -> _NumPyArray:
        return np.transpose(result.grad, np.argsort(axes))

    return result, compute_grad


@create_ufunc_op()
def sum(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _NumPyArray:
        grad_shape = list(result.grad.shape)
        if axis is not None and not keepdims:
            axis_tuple = axis if isinstance(axis, tuple) else (axis,)
            for ax in axis_tuple:
                grad_shape.insert(ax, 1)

        grad = np.reshape(result.grad, grad_shape)
        return grad

    return result, compute_grad


@create_ufunc_op()
def trace(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.trace(self.data))

    def compute_grad() -> _NumPyArray:
        grad = np.zeros_like(self.data)
        np.fill_diagonal(grad, 1)
        return grad * result.grad

    return result, compute_grad


@create_ufunc_op()
def mean(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _NumPyArray:
        if axis is None:
            count = self.data.size
        else:
            axis_tuple = axis if isinstance(axis, tuple) else (axis,)
            count = np.prod([self.shape[ax] for ax in axis_tuple])

        grad_shape = list(result.grad.shape)
        if axis is not None and not keepdims:
            for ax in axis_tuple:
                grad_shape.insert(ax, 1)

        grad = np.reshape(result.grad, grad_shape)
        return grad / count

    return result, compute_grad


@create_ufunc_op()
def var(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(np.var(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _NumPyArray:
        if axis is None:
            count = self.data.size
        else:
            axis_tuple = axis if isinstance(axis, tuple) else (axis,)
            count = np.prod([self.data.shape[ax] for ax in axis_tuple])

        mean_val = np.mean(self.data, axis=axis, keepdims=True)
        grad = (2 / count) * (self.data - mean_val) * result.grad

        if axis is not None and not keepdims:
            grad_shape = list(result.grad.shape)
            for ax in sorted(axis_tuple):
                grad_shape.insert(ax, 1)
            grad = np.reshape(grad, grad_shape)

        return grad

    return result, compute_grad
