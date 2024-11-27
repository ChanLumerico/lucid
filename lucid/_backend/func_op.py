import functools
from typing import Callable, Tuple

import numpy as np

import lucid
from lucid._tensor import Tensor
from lucid.types import _NumPyArray, _ArrayOrScalar

_GradFuncType = Callable[[None], _NumPyArray | Tuple[_NumPyArray, ...]]

_ReturnGradFuncPair = Tuple[Tensor, _GradFuncType]

_FuncOpReturnType = _ReturnGradFuncPair | Tuple[_ReturnGradFuncPair, ...]


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


def create_func_op(n_in: int | None, n_ret: int, has_gradient: bool = True) -> callable:

    def decorator(func: Callable[..., _FuncOpReturnType]) -> callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[Tensor, ...]:
            tensors: Tuple[Tensor, ...] = tuple()
            requires_grad = False

            if n_in is None:
                tensor_args = args
            else:
                if len(args) < n_in:
                    raise ValueError(
                        f"Expected at least {n_in} tensor arguments, got {len(args)}"
                    )
                tensor_args = args[:n_in]

            for arg in tensor_args:
                tensor = _check_is_tensor(arg)
                tensors += (tensor,)
                requires_grad = requires_grad or tensor.requires_grad

            non_tensor_args = args[n_in:] if n_in is not None else ()
            new_args = (*tensors, *non_tensor_args)

            func_return_pairs = func(*new_args, **kwargs)

            if n_ret == 1:
                func_return_pairs = (func_return_pairs,)

            results: Tuple[Tensor, ...] = tuple()
            for result, compute_grad in func_return_pairs:
                result.requires_grad = requires_grad and has_gradient
                results += (result,)

                def _backward_op(*, _func: Callable = compute_grad) -> None:
                    grads = _func()
                    if n_in == 1 or not isinstance(grads, tuple):
                        grads = (grads,)

                    if len(grads) != len(tensors):
                        raise ValueError(
                            f"Expected {len(tensors)} gradients, got {len(grads)}."
                        )

                    for tensor, grad in zip(tensors, grads):
                        new_grad = _match_grad_shape(tensor.data, grad)
                        _set_tensor_grad(tensor, new_grad)

                if not lucid.grad_enabled():
                    continue

                if result.requires_grad:
                    result._backward_op = _backward_op
                    result._prev = tensors

            return results if n_ret > 1 else results[0]

        return wrapper

    return decorator


def create_bfunc_op(has_gradient: bool = True) -> callable:
    return create_func_op(n_in=2, n_ret=1, has_gradient=has_gradient)


def create_ufunc_op(has_gradient: bool = True) -> callable:
    return create_func_op(n_in=1, n_ret=1, has_gradient=has_gradient)


def create_mfunc_op(has_gradient: bool = True) -> callable:
    return create_func_op(n_in=None, n_ret=1, has_gradient=has_gradient)
