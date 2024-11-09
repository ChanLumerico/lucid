import numpy as np

from lucid._tensor import Tensor
from lucid._func._common import create_bfunc_op
from lucid.types import _ArrayOrScalar


@create_bfunc_op()
def _add(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(self.data + other.data)

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1, 1

    return result, compute_grad


@create_bfunc_op()
def _sub(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(self.data - other.data)

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1, -1

    return result, compute_grad


@create_bfunc_op()
def _mul(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(self.data * other.data)

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return other.data, self.data

    return result, compute_grad


@create_bfunc_op()
def _truediv(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(self.data / other.data)

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1 / other.data, -self.data / (other.data**2)

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _equal(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor((self.data == other.data).astype(self.dtype))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _not_equal(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor((self.data != other.data).astype(self.dtype))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _greater(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor((self.data > other.data).astype(self.dtype))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _greater_or_equal(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor((self.data >= other.data).astype(self.dtype))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _less(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor((self.data < other.data).astype(self.dtype))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op(has_gradient=False)
def _less_or_equal(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor((self.data <= other.data).astype(self.dtype))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op()
def minimum(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.minimum(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        self_grad = (self.data <= other.data).astype(self.dtype)
        other_grad = (self.data > other.data).astype(other.dtype)
        return self_grad, other_grad

    return result, compute_grad


@create_bfunc_op()
def maximum(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.maximum(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        self_grad = (self.data >= other.data).astype(self.dtype)
        other_grad = (other.data > self.data).astype(other.dtype)
        return self_grad, other_grad

    return result, compute_grad


@create_bfunc_op()
def power(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.power(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        self_grad = other.data * np.power(self.data, other.data - 1)
        other_grad = np.power(self.data, other.data) * np.log(self.data)
        return self_grad, other_grad

    return result, compute_grad


@create_bfunc_op()
def dot(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.dot(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return other.data, self.data

    return result, compute_grad


@create_bfunc_op()
def inner(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.inner(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return other.data, self.data

    return result, compute_grad


@create_bfunc_op()
def outer(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.outer(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        self_grad = np.reshape(other.data, (1, other.data.size))
        other_grad = np.reshape(self.data, (self.data.size, 1))
        return self_grad, other_grad

    return result, compute_grad


@create_bfunc_op()
def matmul(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(np.matmul(self.data, other.data))

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        grad_x = np.matmul(other.data.T, result.grad)
        grad_y = np.matmul(self.data.T, result.grad)
        return grad_x, grad_y

    return result, compute_grad


_radd: callable = lambda self, other: _add(self, other)
_rsub: callable = lambda self, other: _sub(other, self)
_rmul: callable = lambda self, other: _mul(self, other)
_rtruediv: callable = lambda self, other: _truediv(other, self)
