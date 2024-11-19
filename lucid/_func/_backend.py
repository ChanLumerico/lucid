import functools
from typing import Any

import numpy as np

import lucid
from lucid._tensor import Tensor
from lucid.types import _NumPyArray, _ArrayOrScalar


def _set_tensor_grad(tensor: Tensor, grad: _NumPyArray) -> None:
    if tensor.requires_grad:
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad = tensor.grad + grad


def _check_is_tensor(any: Tensor | _ArrayOrScalar) -> Tensor:
    if not isinstance(any, Tensor):
        return Tensor(any)
    return any


def _match_grad_shape(data: _NumPyArray, grad: _NumPyArray) -> _NumPyArray:
    if data.shape == grad.shape:
        return grad

    if data.size == grad.size:
        reshaped_grad = grad
    elif data.size < grad.size:
        axis = []
        if data.ndim == 0:
            axis.extend(range(grad.ndim))
        else:
            for ax in range(data.ndim):
                if data.shape[ax] != grad.shape[ax] and data.shape[ax] == 1:
                    axis.append(ax)

        reshaped_grad = np.sum(grad, axis=tuple(axis)).reshape(data.shape)
    else:
        reshaped_grad = np.broadcast_to(grad, data.shape)

    return reshaped_grad


# TODO: Test this generalized decorator factory
def create_func_op(n_in: int, n_ret: int, has_gradient: bool = True) -> callable:

    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> tuple[Tensor, ...]:
            tensors: list[Tensor] = []
            requires_grad = False

            for arg in args[:n_in]:
                tensor = _check_is_tensor(arg)
                tensors.append(tensor)
                requires_grad = requires_grad or tensor.requires_grad

            if len(tensors) != n_in:
                return ValueError(f"Number of input tensors foes not match.")

            new_args = (*tensors, *args[n_in:])
            results, compute_grad = func(*new_args, **kwargs)

            if len(results) != n_ret:
                return ValueError(f"Number of returned tensors does not match.")

            for result in results:
                result.requires_grad = requires_grad and has_gradient

            if not lucid.grad_enabled():
                return results

            def _backward_op() -> None:
                grads: tuple[_NumPyArray] = compute_grad()
                for i in range(n_in):
                    new_grad = _match_grad_shape(tensors[i].data, grads[i])
                    _set_tensor_grad(tensor[i], new_grad)

            for result in results:
                result._backward_op = _backward_op
                result._prev = tensors

            return results

        return wrapper

    return decorator


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

            if not lucid.grad_enabled():
                return result

            def _backward_op() -> None:
                self_grad, other_grad = compute_grad()
                self_grad = _match_grad_shape(self.data, self_grad)
                other_grad = _match_grad_shape(other.data, other_grad)

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

            if not lucid.grad_enabled():
                return result

            def _backward_op() -> None:
                self_grad = compute_grad()
                self_grad = _match_grad_shape(self.data, self_grad)

                _set_tensor_grad(self, self_grad)

            result._backward_op = _backward_op
            result._prev = [self]

            return result

        return wrapper

    return decorator
