import functools
from typing import Any

from lucid._tensor import Tensor
from lucid.types import _NumPyArray


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


def create_bfunc_op(has_gradient: bool = True) -> callable:

    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(self: Any, other: Any, *args, **kwargs) -> Tensor:
            self = _check_is_tensor(self)
            other = _check_is_tensor(other)

            result, compute_grad = func(self, other, *args, **kwargs)
            result.requires_grad = self.requires_grad or other.requires_grad

            if not has_gradient:
                result.requires_grad = False

            def _backward_op() -> None:
                self_grad, other_grad = compute_grad()
                _set_tensor_grad(self, self_grad)
                _set_tensor_grad(other, other_grad)

            result._backward_op = _backward_op
            result._prev = [self, other]

            return result

        return wrapper

    return decorator


def create_ufunc_op(has_gradient: bool = True) -> callable:

    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(self: Any, *args, **kwargs) -> Tensor:
            self = _check_is_tensor(self)

            result, compute_grad = func(self, *args, **kwargs)
            result.requires_grad = self.requires_grad

            if not has_gradient:
                result.requires_grad = False

            def _backward_op() -> None:
                self_grad = compute_grad()
                _set_tensor_grad(self, self_grad)

            result._backward_op = _backward_op
            result._prev = [self]

            return result

        return wrapper

    return decorator
