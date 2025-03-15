from functools import partial
from types import ModuleType
from typing import Optional, Literal
import numpy as np

from lucid._tensor import Tensor
from lucid.types import _Scalar

from lucid._backend.core import (
    operation,
    unary_func_op,
    _FuncOpReturnType,
    _GradFuncType,
)
from lucid._backend.metal import mx

# tmp
from lucid.types import _NumPyArray
from lucid._backend.metal import _MLXArray


class _pow(operation):
    def __init__(self, exp: _Scalar) -> None:
        self.exp = exp
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data**self.exp)
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data**self.exp)
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return (self.exp * a.data ** (self.exp - 1)) * self.result.grad


class _neg(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(-a.data)
        return self.result, self.compute_grad

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(-a.data)
        return self.result, self.compute_grad

    def compute_grad(self) -> _GradFuncType:
        return -self.result.grad


class exp(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.exp(a.data))
        return self.result, self.compute_grad

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.exp(a.data))
        return self.result, self.compute_grad

    def compute_grad(self) -> _GradFuncType:
        return self.result.data * self.result.grad


class log(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.log(a.data))
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.log(a.data))
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return (1 / a.data) * self.result.grad


class log2(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.log2(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.log2(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return (1 / (a.data * lib_.log(2))) * self.result.grad


class sqrt(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.sqrt(a.data))
        return self.result, self.compute_grad

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.sqrt(a.data))
        return self.result, self.compute_grad

    def compute_grad(self) -> _GradFuncType:
        return (0.5 / self.result.data) * self.result.grad


class sin(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.sin(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.sin(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return lib_.cos(a.data) * self.result.grad


class cos(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.cos(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.cos(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return -lib_.sin(a.data) * self.result.grad


class tan(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.tan(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.tan(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return (1 / (lib_.cos(a.data) ** 2)) * self.result.grad


class arcsin(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.arcsin(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.arcsin(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return (1 / lib_.sqrt(1 - a.data**2)) * self.result.grad


class arccos(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.arccos(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.arccos(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return (-1 / lib_.sqrt(1 - a.data**2)) * self.result.grad


class arctan(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.arctan(a.data))
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.arctan(a.data))
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return (1 / (1 + a.data**2)) * self.result.grad


class sinh(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> Tensor:
        self.result = Tensor(np.sinh(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> Tensor:
        self.result = Tensor(mx.sinh(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return lib_.cosh(a.data) * self.result.grad


class cosh(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> Tensor:
        self.result = Tensor(np.cosh(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> Tensor:
        self.result = Tensor(mx.cosh(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return lib_.sinh(a.data) * self.result.grad


class tanh(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> Tensor:
        self.result = Tensor(np.tanh(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> Tensor:
        self.result = Tensor(mx.tanh(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return (1 - lib_.tanh(a.data) ** 2) * self.result.grad


class clip(operation):
    def __init__(self, min_value: float | None, max_value: float) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.clip(a.data, self.min_value, self.max_value))
        return self.result, partial(self.compute_grad_cpu, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.clip(a.data, self.min_value, self.max_value))
        return self.result, partial(self.compute_grad_gpu, a=a)

    def compute_grad_cpu(self, a: Tensor) -> _GradFuncType:
        grad = np.ones_like(a.data)
        grad[a.data < self.min_value] = 0
        grad[a.data > self.max_value] = 0

        return grad * self.result.grad

    def compute_grad_gpu(self, a: Tensor) -> _GradFuncType:
        grad = mx.ones_like(a.data)
        grad = mx.where(a.data < self.min_value, 0, grad)
        grad = mx.where(a.data > self.max_value, 0, grad)

        return grad * self.result.grad


class abs(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.abs(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.abs(a.data))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        return lib_.where(a.data >= 0, 1, -1) * self.result.grad


class sign(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op(has_gradient=False)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.sign(a.data))
        return self.result, partial(self.compute_grad, lib_=np)

    @unary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.sign(a.data))
        return self.result, partial(self.compute_grad, lib_=mx)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0)


class reciprocal(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(1 / a.data)
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(1 / a.data)
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return (-1 / (a.data**2)) * self.result.grad


class square(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.square(a.data))
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.square(a.data))
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return 2 * a.data * self.result.grad


class cube(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data**3)
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data**3)
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return 3 * a.data**2 * self.result.grad


class _T(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.T)
        return self.result, self.compute_grad

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.T)
        return self.result, self.compute_grad

    def compute_grad(self) -> _GradFuncType:
        return self.result.grad


class _mT(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.mT)
        return self.result, self.compute_grad_cpu

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.swapaxes(a.data, -1, -2))
        return self.result, self.compute_grad_gpu

    def compute_grad_cpu(self) -> _GradFuncType:
        return self.result.grad.mT

    def compute_grad_gpu(self) -> _GradFuncType:
        return mx.swapaxes(self.result.grad, -1, -2)


# TODO: Continue from here


def _get_transpose_axes(axes: Optional[list[int]], ndim: int) -> list:
    if axes is None:
        axes = list(reversed(range(ndim)))
    return axes


@unary_func_op()
def transpose(self: Tensor, axes: Optional[list[int]] = None) -> _FuncOpReturnType:
    axes = _get_transpose_axes(axes, self.ndim)
    result = Tensor(np.transpose(self.data, axes))

    def compute_grad() -> _NumPyArray:
        return np.transpose(result.grad, np.argsort(axes))

    return result, compute_grad


@unary_func_op(device="gpu")
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


@unary_func_op()
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


@unary_func_op(device="gpu")
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


@unary_func_op()
def trace(self: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.trace(self.data))

    def compute_grad() -> _NumPyArray:
        grad = np.zeros_like(self.data)
        np.fill_diagonal(grad, 1)
        return grad * result.grad

    return result, compute_grad


@unary_func_op(device="gpu")
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


@unary_func_op()
def mean(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _NumPyArray:
        return _mean_grad_unified(axis, keepdims, self, result, lib=np)

    return result, compute_grad


@unary_func_op(device="gpu")
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


@unary_func_op()
def var(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(np.var(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _NumPyArray:
        return _var_grad_unified(axis, keepdims, self, result, lib=np)

    return result, compute_grad


@unary_func_op(device="gpu")
def var_gpu(
    self: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> _FuncOpReturnType:
    result = Tensor(mx.var(self.data, axis=axis, keepdims=keepdims))

    def compute_grad() -> _MLXArray:
        return _var_grad_unified(axis, keepdims, self, result, lib=mx)

    return result, compute_grad


def _min_or_max_grad_unified(
    axis: int | tuple[int] | None,
    keepdims: bool,
    self: Tensor,
    result: Tensor,
    lib: ModuleType,
) -> _NumPyArray | _MLXArray:
    grad = result.grad
    if not keepdims and axis is not None:
        if isinstance(axis, tuple):
            for ax in sorted(axis):
                grad = lib.expand_dims(grad, axis=ax)
        else:
            grad = lib.expand_dims(grad, axis=axis)

    if keepdims:
        result_expanded = result.data
    else:
        if axis is None:
            result_expanded = result.data.reshape((1,) * self.data.ndim)
        else:
            if isinstance(axis, tuple):
                result_expanded = result.data
                for ax in sorted(axis):
                    result_expanded = lib.expand_dims(result_expanded, axis=ax)
            else:
                result_expanded = lib.expand_dims(result.data, axis=axis)

    mask = self.data == result_expanded
    counts = lib.sum(mask, axis=axis, keepdims=True)
    counts = lib.where(counts == 0, 1, counts)

    return mask * grad / counts


@unary_func_op()
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
        return _min_or_max_grad_unified(axis, keepdims, self, result, lib=np)

    return result, compute_grad


@unary_func_op(device="gpu")
def _min_or_max_gpu(
    self: Tensor,
    mode: Literal["min", "max"],
    axis: int | tuple[int] | None = None,
    keepdims: bool = False,
) -> _FuncOpReturnType:
    if mode == "max":
        data = mx.max(self.data, axis=axis, keepdims=keepdims)
    else:
        data = mx.min(self.data, axis=axis, keepdims=keepdims)
    result = Tensor(data)

    def compute_grad() -> _MLXArray:
        return _min_or_max_grad_unified(axis, keepdims, self, result, lib=mx)

    return result, compute_grad


@unary_func_op()
def swapaxes(self: Tensor, axis1: int, axis2: int) -> Tensor:
    result = Tensor(self.data.swapaxes(axis1, axis2))

    def compute_grad() -> _NumPyArray:
        return result.grad.swapaxes(axis1, axis2)

    return result, compute_grad


@unary_func_op(device="gpu")
def swapaxes_gpu(self: Tensor, axis1: int, axis2: int) -> Tensor:
    result = Tensor(mx.swapaxes(self.data, axis1, axis2))

    def compute_grad() -> _MLXArray:
        return mx.swapaxes(result.grad, axis1, axis2)

    return result, compute_grad
