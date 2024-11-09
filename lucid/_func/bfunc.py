import numpy as np

from lucid.tensor import Tensor, _ArrayOrScalar
from lucid._func._common import create_bfunc_op


@create_bfunc_op
def add(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        self.data + other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1, 1

    return result, compute_grad


@create_bfunc_op
def sub(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        self.data - other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1, -1

    return result, compute_grad


@create_bfunc_op
def mul(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        self.data * other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return other.data, self.data

    return result, compute_grad


@create_bfunc_op
def truediv(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        self.data / other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1 / other.data, -self.data / (other.data**2)

    return result, compute_grad


@create_bfunc_op
def equal(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        (self.data == other.data).astype(self.dtype),
        requires_grad=False,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op
def greater(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        (self.data > other.data).astype(self.dtype),
        requires_grad=False,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op
def less(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        (self.data < other.data).astype(self.dtype),
        requires_grad=False,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 0, 0

    return result, compute_grad


@create_bfunc_op
def minimum(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        np.minimum(self.data, other.data),
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        self_grad = (self.data <= other.data).astype(self.dtype)
        other_grad = (self.data > other.data).astype(other.dtype)
        return self_grad, other_grad

    return result, compute_grad


@create_bfunc_op
def maximum(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        np.maximum(self.data, other.data),
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        self_grad = (self.data >= other.data).astype(self.dtype)
        other_grad = (other.data > self.data).astype(other.dtype)
        return self_grad, other_grad

    return result, compute_grad


@create_bfunc_op
def power(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        np.power(self.data, other.data),
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        self_grad = other.data * np.power(self.data, other.data - 1)
        other_grad = np.power(self.data, other.data) * np.log(self.data)
        return self_grad, other_grad

    return result, compute_grad


radd: callable = lambda self, other: add(self, other)
rsub: callable = lambda self, other: sub(other, self)
rmul: callable = lambda self, other: mul(self, other)
rtruediv: callable = lambda self, other: truediv(other, self)
