from lucid._tensor import Tensor

from lucid._backend.core import create_bfunc_op, _FuncOpReturnType
from lucid._backend.metal import mx, _MLXArray


@create_bfunc_op(device="gpu")
def _add(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.add(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return result.grad, result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _sub(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.subtract(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return result.grad, -result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _mul(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.multiply(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return other.data * result.grad, self.data * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _truediv(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.divide(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            (1 / other.data) * result.grad,
            (-self.data / (other.data**2)) * result.grad,
        )

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _equal(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data == other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _not_equal(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data != other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _greater(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data > other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _greater_or_equal(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data >= other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _less(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data < other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(has_gradient=False, device="gpu")
def _less_or_equal(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor((self.data <= other.data).astype(self.dtype))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return mx.array(0.0), mx.array(0.0)

    return result, compute_grad


@create_bfunc_op(device="gpu")
def minimum(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.minimum(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        self_grad = (self.data <= other.data).astype(self.dtype)
        other_grad = (self.data > other.data).astype(other.dtype)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def maximum(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.maximum(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        self_grad = (self.data >= other.data).astype(self.dtype)
        other_grad = (other.data > self.data).astype(other.dtype)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def power(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.power(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        self_grad = other.data * mx.power(self.data, other.data - 1)
        other_grad = mx.power(self.data, other.data) * mx.log(self.data)

        return self_grad * result.grad, other_grad * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def dot(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    if self.ndim != 1 or other.ndim != 1:
        raise ValueError(f"Only 1D dot product is supported for Metal backend.")

    result = Tensor(mx.sum(self.data * other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return other.data * result.grad, self.data * result.grad

    return result, compute_grad


@create_bfunc_op(device="gpu")
def inner(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.inner(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.tensordot(result.grad, other.data, axes=((-1,), (-1,))),
            mx.tensordot(self.data, result.grad, axes=((-1,), (-1,))),
        )

    return result, compute_grad


@create_bfunc_op(device="gpu")
def outer(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.outer(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.tensordot(result.grad, other.data, axes=((1,), (0,))),
            mx.tensordot(self.data, result.grad, axes=((0,), (0,))),
        )

    return result, compute_grad


@create_bfunc_op(device="gpu")
def _matmul(self: Tensor, other: Tensor) -> _FuncOpReturnType:
    result = Tensor(mx.matmul(self.data, other.data))

    def compute_grad() -> tuple[_MLXArray, _MLXArray]:
        return (
            mx.matmul(result.grad, mx.swapaxes(other.data, -2, -1)),
            mx.matmul(mx.swapaxes(self.data, -2, -1), result.grad),
        )

    return result, compute_grad
