import functools
from typing import Any

from lucid.tensor import Tensor, _NumPyArray


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


def create_bfunc_op(func: callable) -> callable:

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


def create_ufunc_op(func: callable) -> callable:

    @functools.wraps(func)
    def wrapper(self: Any, *args, **kwargs) -> Tensor:
        self = _check_is_tensor(self)

        result, compute_grad = func(self, *args, **kwargs)

        def _backward_op() -> None:
            self_grad = compute_grad()

            self_grad_chain = self_grad
            if result.grad is not None:
                self_grad_chain = self_grad * result.grad

            _set_tensor_grad(self, self_grad_chain)

        result._backward_op = _backward_op
        result._prev = [self]

        return result

    return wrapper
