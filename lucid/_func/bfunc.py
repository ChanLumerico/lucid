import functools
from typing import Any

import numpy as np

from lucid.tensor import Tensor, _NumPyArray, _ArrayOrScalar


def _set_tensor_grad(tensor: Tensor, grad: _NumPyArray) -> None:
    if tensor.requires_grad:
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad += grad


def _check_is_tensor(any: Any) -> Tensor:
    if not isinstance(any, Tensor):
        return Tensor(any)
    return any


def _create_bfunc_op(func: callable) -> callable:

    @functools.wraps(func)
    def wrapper(self: Any, other: Any, *args, **kwargs) -> Tensor:
        self = _check_is_tensor(self)
        other = _check_is_tensor(other)

        result, compute_grad = func(self, other, *args, **kwargs)

        def _backward_op() -> None:
            self_grad, other_grad = compute_grad()

            self_grad_chain = self_grad
            other_grad_chain = other_grad

            if result.grad is not None:
                self_grad_chain = self_grad * result.grad
                other_grad_chain = other_grad * result.grad

            _set_tensor_grad(self, self_grad_chain)
            _set_tensor_grad(other, other_grad_chain)

        result._backward_op = _backward_op
        result._prev = [self, other]

        return result

    return wrapper


@_create_bfunc_op
def add(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        self.data + other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1, 1

    return result, compute_grad


@_create_bfunc_op
def sub(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        self.data - other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1, -1

    return result, compute_grad


@_create_bfunc_op
def mul(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        self.data * other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return other.data, self.data

    return result, compute_grad


@_create_bfunc_op
def truediv(self: Tensor, other: Tensor) -> tuple[Tensor, callable]:
    result = Tensor(
        self.data / other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def compute_grad() -> tuple[_ArrayOrScalar, _ArrayOrScalar]:
        return 1 / other.data, -self.data / (other.data**2)

    return result, compute_grad


@_create_bfunc_op
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


radd: callable = lambda self, other: add(self, other)
rsub: callable = lambda self, other: sub(other, self)
rmul: callable = lambda self, other: mul(self, other)
rtruediv: callable = lambda self, other: truediv(other, self)
