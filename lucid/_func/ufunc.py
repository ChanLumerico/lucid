import functools
from typing import Any

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


def _create_ufunc_op(func: callable) -> callable:

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


@_create_ufunc_op
def pow(self: Tensor, exp: int | float) -> Tensor:
    result = Tensor(self.data**exp, requires_grad=self.requires_grad)

    def compute_grad() -> _ArrayOrScalar:
        return exp * self.data ** (exp - 1)

    return result, compute_grad
