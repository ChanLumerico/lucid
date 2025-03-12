import numpy as np

from lucid._tensor import Tensor
from lucid._backend.core import create_bfunc_op, _FuncOpReturnType
from lucid._backend.metal import mx, _MLXArray

from lucid.types import _NumPyArray


@create_bfunc_op()
def _add(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(self.data + other.data)

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return result.grad, result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _add_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.add(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return result.grad, result.grad

    return result, compute_grad


@create_bfunc_op()
def _sub(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(self.data - other.data)

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return result.grad, -result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _sub_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.subtract(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return result.grad, -result.grad

    return result, compute_grad


@create_bfunc_op()
def _mul(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(self.data * other.data)

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return other.data * result.grad, self.data * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _mul_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.multiply(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return other.data * result.grad, self.data * result.grad

    return result, compute_grad


@create_bfunc_op()
def _truediv(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(self.data / other.data)

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            (1 / other.data) * result.grad,
            (-self.data / (other.data**2)) * result.grad,
        )

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _truediv_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.divide(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            (1 / other.data) * result.grad,
            (-self.data / (other.data**2)) * result.grad,
        )

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _equal(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data == other.data).astype(self.dtype))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return np.array(0.0), np.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _equal_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data == other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _not_equal(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data != other.data).astype(self.dtype))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return np.array(0.0), np.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _not_equal_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data != other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _greater(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data > other.data).astype(self.dtype))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return np.array(0.0), np.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _greater_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data > other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _greater_or_equal(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data >= other.data).astype(self.dtype))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return np.array(0.0), np.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _greater_or_equal_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data >= other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _less(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data < other.data).astype(self.dtype))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return np.array(0.0), np.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _less_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data < other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _less_or_equal(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data <= other.data).astype(self.dtype))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return np.array(0.0), np.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _less_or_equal_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data <= other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op()
def minimum(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.minimum(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        self_grad = (self.data <= other.data).astype(self.dtype)
        other_grad = (self.data > other.data).astype(other.dtype)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def minimum_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.minimum(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        self_grad = (self.data <= other.data).astype(self.dtype)
        other_grad = (self.data > other.data).astype(other.dtype)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op()
def maximum(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.maximum(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        self_grad = (self.data >= other.data).astype(self.dtype)
        other_grad = (other.data > self.data).astype(other.dtype)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def maximum_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.maximum(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        self_grad = (self.data >= other.data).astype(self.dtype)
        other_grad = (other.data > self.data).astype(other.dtype)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op()
def power(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.power(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        self_grad = other.data * np.power(self.data, other.data - 1)
        other_grad = np.power(self.data, other.data) * np.log(self.data)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def power_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.power(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        self_grad = other.data * mx.power(self.data, other.data - 1)
        other_grad = mx.power(self.data, other.data) * mx.log(self.data)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op()
def dot(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.dot(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            result.grad.dot(other.data.mT),
            self.data.mT.dot(result.grad),
        )

    return result, compute_grad


@create_bfunc_op(device="gpu")
def dot_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    if self.ndim != 1 or other.ndim != 1:
        raise ValueError(f"Only 1D dot product is supported for Metal backend.")

    result = Tensor(mx.sum(self.data * other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return other.data * result.grad, self.data * result.grad

    return result, compute_grad


@create_bfunc_op()
def inner(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.inner(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            np.tensordot(result.grad, other.data, axes=(-1, -1)),
            np.tensordot(self.data, result.grad, axes=(-1, -1)),
        )

    return result, compute_grad


@create_bfunc_op(device="gpu")
def inner_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.inner(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.tensordot(result.grad, other.data, axes=((-1,), (-1,))),
            mx.tensordot(self.data, result.grad, axes=((-1,), (-1,))),
        )

    return result, compute_grad


@create_bfunc_op()
def outer(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.outer(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            np.tensordot(result.grad, other.data, axes=(1, 0)),
            np.tensordot(result.grad, self.data, axes=(0, 0)),
        )

    return result, compute_grad


@create_bfunc_op(device="gpu")
def outer_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.outer(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.tensordot(result.grad, other.data, axes=((1,), (0,))),
            mx.tensordot(self.data, result.grad, axes=((0,), (0,))),
        )

    return result, compute_grad


@create_bfunc_op()
def _matmul(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(np.matmul(self.data, other.data))

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        return (
            np.matmul(result.grad, other.data.mT),
            np.matmul(self.data.mT, result.grad),
        )

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _matmul_gpu(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.matmul(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.matmul(result.grad, mx.swapaxes(other.data, -2, -1)),
            mx.matmul(mx.swapaxes(self.data, -2, -1), result.grad),
        )

    return result, compute_grad
