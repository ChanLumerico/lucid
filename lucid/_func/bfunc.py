from functools import partial
from types import ModuleType
import numpy as np

from lucid._tensor import Tensor
from lucid._backend.core import (
    operation,
    binary_func_op,
    _FuncOpReturnType,
    _GradFuncType,
)
from lucid._backend.metal import mx, _MLXArray

from lucid.types import _NumPyArray


class add(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data + b.data)
        return self.result, self.compute_grad

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.add(a.data, b.data))
        return self.result, self.compute_grad

    def compute_grad(self) -> _GradFuncType:
        return self.result.grad, self.result.grad


class sub(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data - b.data)
        return self.result, self.compute_grad

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.subtract(a.data, b.data))
        return self.result, self.compute_grad

    def compute_grad(self) -> _GradFuncType:
        return self.result.grad, -self.result.grad


class multiply(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data * b.data)
        return self.result, partial(self.compute_grad, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.multiply(a.data, b.data))
        return self.result, partial(self.compute_grad, a=a, b=b)

    def compute_grad(self, a: Tensor, b: Tensor) -> _GradFuncType:
        return b.data * self.result.grad, a.data * self.result.grad


class truediv(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data / b.data)
        return self.result, partial(self.compute_grad, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.divide(a.data, b.data))
        return self.result, partial(self.compute_grad, a=a, b=b)

    def compute_grad(self, a: Tensor, b: Tensor) -> _GradFuncType:
        return (
            (1 / b.data) * self.result.grad,
            (-a.data / (b.data**2)) * self.result.grad,
        )


class _equal(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> Tensor:
        self.result = Tensor((a.data == b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> Tensor:
        self.result = Tensor((a.data == b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=mx)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _not_equal(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> Tensor:
        self.result = Tensor((a.data != b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> Tensor:
        self.result = Tensor((a.data != b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _greater(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data > b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data > b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _greater_or_equal(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data >= b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data >= b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _less(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data < b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data < b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class _less_or_equal(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op(has_gradient=False)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data <= b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    @binary_func_op(has_gradient=False, device="gpu")
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor((a.data <= b.data).astype(a.dtype))
        return self.result, partial(self.compute_grad, lib=np)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0), lib_.array(0.0)


class minimum(operation):
    def __init__(self) -> None:
        super().__init__()

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.minimum(a.data, b.data))
        return self.result, partial(self.compute_grad, a=a, b=b)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.minimum(a.data, b.data))
        return self.result, partial(self.compute_grad, a=a, b=b)

    def compute_grad(self, a: Tensor, b: Tensor) -> _GradFuncType:
        a_grad = (a.data <= b.data).astype(a.dtype)
        b_grad = (a.data > b.data).astype(b.dtype)

        return a_grad * self.result.grad, b_grad * self.result.grad


# TODO: Continue from here


@binary_func_op()
def maximum(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.maximum(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        self_grad = (self.data >= other.data).astype(self.dtype)
        other_grad = (other.data > self.data).astype(other.dtype)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@binary_func_op(device="gpu")
def maximum_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.maximum(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        self_grad = (self.data >= other.data).astype(self.dtype)
        other_grad = (other.data > self.data).astype(other.dtype)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@binary_func_op()
def power(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.power(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        self_grad = other.data * np.power(self.data, other.data - 1)
        other_grad = np.power(self.data, other.data) * np.log(self.data)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@binary_func_op(device="gpu")
def power_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.power(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        self_grad = other.data * mx.power(self.data, other.data - 1)
        other_grad = mx.power(self.data, other.data) * mx.log(self.data)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@binary_func_op()
def dot(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.dot(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            result.grad.dot(other.data.mT),
            self.data.mT.dot(result.grad),
        )

    return result, compute_grad


@binary_func_op(device="gpu")
def dot_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    if self.ndim != 1 or other.ndim != 1:
        raise ValueError(f"Only 1D dot product is supported for Metal backend.")

    result = Tensor(mx.sum(self.data * other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return other.data * result.grad, self.data * result.grad

    return result, compute_grad


@binary_func_op()
def inner(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.inner(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            np.tensordot(result.grad, other.data, axes=(-1, -1)),
            np.tensordot(self.data, result.grad, axes=(-1, -1)),
        )

    return result, compute_grad


@binary_func_op(device="gpu")
def inner_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.inner(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.tensordot(result.grad, other.data, axes=((-1,), (-1,))),
            mx.tensordot(self.data, result.grad, axes=((-1,), (-1,))),
        )

    return result, compute_grad


@binary_func_op()
def outer(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.outer(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            np.tensordot(result.grad, other.data, axes=(1, 0)),
            np.tensordot(result.grad, self.data, axes=(0, 0)),
        )

    return result, compute_grad


@binary_func_op(device="gpu")
def outer_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.outer(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.tensordot(result.grad, other.data, axes=((1,), (0,))),
            mx.tensordot(self.data, result.grad, axes=((0,), (0,))),
        )

    return result, compute_grad


@binary_func_op()
def _matmul(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.matmul(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            np.matmul(result.grad, other.data.mT),
            np.matmul(self.data.mT, result.grad),
        )

    return result, compute_grad


@binary_func_op(device="gpu")
def _matmul_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.matmul(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.matmul(result.grad, mx.swapaxes(other.data, -2, -1)),
            mx.matmul(mx.swapaxes(self.data, -2, -1), result.grad),
        )

    return result, compute_grad
