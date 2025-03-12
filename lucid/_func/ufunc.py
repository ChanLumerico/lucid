from typing import Literal, Optional
from types import ModuleType
import numpy as np

from lucid._tensor import Tensor
from lucid._backend.core import create_ufunc_op, _FuncOpReturnType
from lucid._backend.metal import mx, _MLXArray

from lucid.types import _NumPyArray, _Scalar


@create_ufunc_op()
def _pow(self: Tensor, exp: _Scalar) -> _FuncOpReturnType:
    result = Tensor(self.data**exp)

    def compute_grad() -> _NumPyArray:
        return (exp * self.data ** (exp - 1)) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def _pow_gpu(self: Tensor, exp: _Scalar) -> _FuncOpReturnType:
    result = Tensor(self.data**exp)

    def compute_grad() -> _MLXArray:
        return (exp * self.data ** (exp - 1)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def _neg(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(-self.data)

    def compute_grad() -> _NumPyArray:
        return -result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def _neg_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(-self.data)

    def compute_grad() -> _MLXArray:
        return -result.grad

    return result, compute_grad


@create_ufunc_op()
def exp(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.exp(self.data))

    def compute_grad() -> _NumPyArray:
        return result.data * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def exp_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.exp(self.data))

    def compute_grad() -> _MLXArray:
        return result.data * result.grad

    return result, compute_grad


@create_ufunc_op()
def log(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.log(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / self.data) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def log_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.log(self.data))

    def compute_grad() -> _MLXArray:
        return (1 / self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def log2(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.log2(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / (self.data * np.log(2))) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def log2_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.log2(self.data))

    def compute_grad() -> _MLXArray:
        return (1 / (self.data * mx.log(2))) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sqrt(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.sqrt(self.data))

    def compute_grad() -> _NumPyArray:
        return (0.5 / result.data) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def sqrt_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.sqrt(self.data))

    def compute_grad() -> _MLXArray:
        return (0.5 / result.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sin(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.sin(self.data))

    def compute_grad() -> _NumPyArray:
        return np.cos(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def sin_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.sin(self.data))

    def compute_grad() -> _MLXArray:
        return mx.cos(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def cos(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.cos(self.data))

    def compute_grad() -> _NumPyArray:
        return -np.sin(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def cos_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.cos(self.data))

    def compute_grad() -> _MLXArray:
        return -mx.sin(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def tan(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.tan(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / (np.cos(self.data) ** 2)) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def tan_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.tan(self.data))

    def compute_grad() -> _MLXArray:
        return (1 / (mx.cos(self.data) ** 2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arcsin(self: Tensor) -> Tensor:
    result = Tensor(np.arcsin(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / np.sqrt(1 - self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def arcsin_gpu(self: Tensor) -> Tensor:
    result = Tensor(mx.arcsin(self.data))

    def compute_grad() -> _MLXArray:
        return (1 / mx.sqrt(1 - self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arccos(self: Tensor) -> Tensor:
    result = Tensor(np.arccos(self.data))

    def compute_grad() -> _NumPyArray:
        return (-1 / np.sqrt(1 - self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def arccos_gpu(self: Tensor) -> Tensor:
    result = Tensor(mx.arccos(self.data))

    def compute_grad() -> _MLXArray:
        return (-1 / mx.sqrt(1 - self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def arctan(self: Tensor) -> Tensor:
    result = Tensor(np.arctan(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 / (1 + self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def arctan_gpu(self: Tensor) -> Tensor:
    result = Tensor(mx.arctan(self.data))

    def compute_grad() -> _MLXArray:
        return (1 / (1 + self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def sinh(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.sinh(self.data))

    def compute_grad() -> _NumPyArray:
        return np.cosh(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def sinh_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.sinh(self.data))

    def compute_grad() -> _MLXArray:
        return mx.cosh(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def cosh(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.cosh(self.data))

    def compute_grad() -> _NumPyArray:
        return np.sinh(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def cosh_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.cosh(self.data))

    def compute_grad() -> _MLXArray:
        return mx.sinh(self.data) * result.grad

    return result, compute_grad


@create_ufunc_op()
def tanh(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.tanh(self.data))

    def compute_grad() -> _NumPyArray:
        return (1 - np.tanh(self.data) ** 2) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def tanh_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.tanh(self.data))

    def compute_grad() -> _MLXArray:
        return (1 - mx.tanh(self.data) ** 2) * result.grad

    return result, compute_grad


@create_ufunc_op()
def clip(self: Tensor, min_value: float | None, max_value: float) -> _FuncOpReturnType:
    result = Tensor(np.clip(self.data, min_value, max_value))

    def compute_grad() -> _NumPyArray:
        grad = np.ones_like(self.data)
        grad[self.data < min_value] = 0
        grad[self.data > max_value] = 0

        return grad * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def clip_gpu(
    self: Tensor, min_value: float | None, max_value: float
) -> _FuncOpReturnType:
    result = Tensor(mx.clip(self.data, min_value, max_value))

    def compute_grad() -> _MLXArray:
        grad = mx.ones_like(self.data)
        grad = mx.where(self.data < min_value, 0, grad)
        grad = mx.where(self.data > max_value, 0, grad)

        return grad * result.grad

    return result, compute_grad


@create_ufunc_op()
def abs(self: Tensor) -> Tensor:
    result = Tensor(np.abs(self.data))

    def compute_grad() -> _NumPyArray:
        return np.where(self.data >= 0, 1, -1) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def abs_gpu(self: Tensor) -> Tensor:
    result = Tensor(mx.abs(self.data))

    def compute_grad() -> _MLXArray:
        return mx.where(self.data >= 0, 1, -1) * result.grad

    return result, compute_grad


@create_ufunc_op(has_gradient=False)
def sign(self: Tensor) -> Tensor:
    result = Tensor(np.sign(self.data))

    def compute_grad() -> _NumPyArray:
        return np.array(0.0)

    return result, compute_grad


@create_ufunc_op(has_gradient=False, device="gpu")
def sign_gpu(self: Tensor) -> Tensor:
    result = Tensor(mx.sign(self.data))

    def compute_grad() -> _MLXArray:
        return mx.array(0.0)

    return result, compute_grad


@create_ufunc_op()
def reciprocal(self: Tensor) -> Tensor:
    result = Tensor(1 / self.data)

    def compute_grad() -> _NumPyArray:
        return (-1 / (self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def reciprocal_gpu(self: Tensor) -> Tensor:
    result = Tensor(1 / self.data)

    def compute_grad() -> _MLXArray:
        return (-1 / (self.data**2)) * result.grad

    return result, compute_grad


@create_ufunc_op()
def square(self: Tensor) -> Tensor:
    result = Tensor(np.square(self.data))

    def compute_grad() -> _NumPyArray:
        return 2 * self.data * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def square_gpu(self: Tensor) -> Tensor:
    result = Tensor(mx.square(self.data))

    def compute_grad() -> _MLXArray:
        return 2 * self.data * result.grad

    return result, compute_grad


@create_ufunc_op()
def cube(self: Tensor) -> Tensor:
    result = Tensor(self.data**3)

    def compute_grad() -> _NumPyArray:
        return 3 * self.data**2 * result.grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def cube_gpu(self: Tensor) -> Tensor:
    result = Tensor(self.data**3)

    def compute_grad() -> _MLXArray:
        return 3 * self.data**2 * result.grad

    return result, compute_grad


@create_ufunc_op()
def _T(self: Tensor) -> Tensor:
    result = Tensor(self.data.T)

    def compute_grad() -> _NumPyArray:
        return result.grad.T

    return result, compute_grad


@create_ufunc_op(device="gpu")
def _T_gpu(self: Tensor) -> Tensor:
    result = Tensor(self.data.T)

    def compute_grad() -> _MLXArray:
        return result.grad.T

    return result, compute_grad


@create_ufunc_op()
def _mT(self: Tensor) -> Tensor:
    result = Tensor(self.data.mT)

    def compute_grad() -> _NumPyArray:
        return result.grad.mT

    return result, compute_grad


@create_ufunc_op(device="gpu")
def _mT_gpu(self: Tensor) -> Tensor:
    result = Tensor(mx.swapaxes(self.data, -1, -2))

    def compute_grad() -> _MLXArray:
        return mx.swapaxes(result.grad, -1, -2)

    return result, compute_grad


def _get_transpose_axes(axes: Optional[list[int]], ndim: int) -> list:
    if axes is None:
        axes = list(reversed(range(ndim)))
    return axes


@create_ufunc_op()
def transpose(self: Tensor, axes: Optional[list[int]] = None) -> _FuncOpReturnType:
    axes = _get_transpose_axes(axes, self.ndim)
    result = Tensor(np.transpose(self.data, axes))

    def compute_grad() -> _NumPyArray:
        return np.transpose(result.grad, np.argsort(axes))

    return result, compute_grad


@create_ufunc_op(device="gpu")
def transpose_gpu(self: Tensor, axes: Optional[list[int]] = None) -> _FuncOpReturnType:
    axes = _get_transpose_axes(axes, self.ndim)
    result = Tensor(mx.transpose(self.data, axes))

    def compute_grad() -> _MLXArray:
        return mx.transpose(result.grad, mx.argsort(mx.array(axes)))

    return result, compute_grad


def _get_sum_grad_shape(
    shape: tuple[int], axis: int | tuple[int], keepdims: bool
) -> tuple[int]:
    grad_shape = list(shape)
    if not keepdims:
        axis_tuple = axis if isinstance(axis, tuple) else (axis,)
        for ax in axis_tuple:
            grad_shape.insert(ax, 1)

    return tuple(grad_shape)


@create_ufunc_op()
def sum(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _NumPyArray:
        if axis is None:
            grad = np.ones_like(self.data) * result.grad
        else:
            grad_shape = _get_sum_grad_shape(result.grad.shape, axis, keepdims)
            grad = np.reshape(result.grad, grad_shape)

        return grad

    return result, compute_grad


@create_ufunc_op(device="gpu")
def sum_gpu(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(mx.sum(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _MLXArray:
        if axis is None:
            grad = mx.ones_like(self.data) * result.grad
        else:
            grad_shape = _get_sum_grad_shape(result.grad.shape, axis, keepdims)
            grad = mx.reshape(result.grad, grad_shape)

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


@create_ufunc_op(device="gpu")
def trace_gpu(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.trace(self.data))

    def compute_grad() -> _MLXArray:
        grad = mx.eye(self.shape[0], dtype=self.dtype)
        return grad * result.grad

    return result, compute_grad


def _mean_grad_unified(
    axis: int | tuple[int] | None,
    keepdims: bool,
    self: Tensor,
    result: Tensor,
    lib: ModuleType,
) -> _NumPyArray | _MLXArray:
    if axis is None:
        count = self.data.size
        grad = lib.ones_like(self.data) * result.grad
    else:
        axis_tuple = axis if isinstance(axis, tuple) else (axis,)
        count = lib.prod(lib.array([self.shape[ax] for ax in axis_tuple]))

        grad_shape = list(result.grad.shape)
        if not keepdims:
            for ax in sorted(axis_tuple):
                grad_shape.insert(ax, 1)

        grad = lib.reshape(result.grad, grad_shape)

    return grad / count


@create_ufunc_op()
def mean(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _NumPyArray:
        return _mean_grad_unified(axis, keepdims, self, result, lib=np)

    return result, compute_grad


@create_ufunc_op(device="gpu")
def mean_gpu(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(mx.mean(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _MLXArray:
        return _mean_grad_unified(axis, keepdims, self, result, lib=mx)

    return result, compute_grad


def _var_grad_unified(
    axis: int | tuple[int] | None,
    keepdims: bool,
    self: Tensor,
    result: Tensor,
    lib: ModuleType,
) -> _NumPyArray | _MLXArray:
    if axis is None:
        count = self.data.size
    else:
        axis_tuple = axis if isinstance(axis, tuple) else (axis,)
        count = lib.prod(lib.array([self.data.shape[ax] for ax in axis_tuple]))

    mean_val = lib.mean(self.data, axis=axis, keepdims=True)
    grad = (2 / count) * (self.data - mean_val) * result.grad

    if axis is not None and not keepdims:
        grad_shape = list(result.grad.shape)
        for ax in sorted(axis_tuple):
            grad_shape.insert(ax, 1)
        grad = lib.reshape(grad, grad_shape)

    return grad


@create_ufunc_op()
def var(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(np.var(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _NumPyArray:
        return _var_grad_unified(axis, keepdims, self, result, lib=np)

    return result, compute_grad


@create_ufunc_op(device="gpu")
def var_gpu(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(mx.var(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _MLXArray:
        return _var_grad_unified(axis, keepdims, self, result, lib=mx)

    return result, compute_grad


@create_ufunc_op()
def _min_or_max(
    self: Tensor,
    mode: Literal["min", "max"],
    axis: int | tuple[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    if mode == "max":
        data = np.max(self.data, axis=axis, keepdims=keepdims)
    else:
        data = np.min(self.data, axis=axis, keepdims=keepdims)
    result = Tensor(data)

    def compute_grad() -> _NumPyArray:
        grad = result.grad
        if not keepdims and axis is not None:
            if isinstance(axis, tuple):
                for ax in sorted(axis):
                    grad = np.expand_dims(grad, axis=ax)
            else:
                grad = np.expand_dims(grad, axis=axis)

        if keepdims:
            result_expanded = result.data
        else:
            if axis is None:
                result_expanded = result.data.reshape((1,) * self.data.ndim)
            else:
                if isinstance(axis, tuple):
                    result_expanded = result.data
                    for ax in sorted(axis):
                        result_expanded = np.expand_dims(result_expanded, axis=ax)
                else:
                    result_expanded = np.expand_dims(result.data, axis=axis)

        mask = self.data == result_expanded
        counts = np.sum(mask, axis=axis, keepdims=True)
        counts = np.where(counts == 0, 1, counts)

        return mask * grad / counts

    return result, compute_grad


@create_ufunc_op()
def swapaxes(self: Tensor, axis1: int, axis2: int) -> Tensor:
    result = Tensor(self.data.swapaxes(axis1, axis2))

    def compute_grad() -> _NumPyArray:
        return result.grad.swapaxes(axis1, axis2)

    return result, compute_grad
