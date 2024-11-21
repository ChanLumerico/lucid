import functools
from typing import Callable, Tuple

import numpy as np

import lucid
from lucid._tensor import Tensor
from lucid.types import _NumPyArray, _ArrayOrScalar

_ResultGradFuncPair = Tuple[Tensor, Callable[[None], Tuple[_NumPyArray, ...]]]
_FuncOpReturnType = _ResultGradFuncPair | Tuple[_ResultGradFuncPair, ...]


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


def create_func_op(n_in: int, n_ret: int, has_gradient: bool = True) -> callable:

    def decorator(func: Callable[..., _FuncOpReturnType]) -> callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> tuple[Tensor, ...]:
            tensors: list[Tensor] = []
            requires_grad = False

            for arg in args[:n_in]:
                tensor = _check_is_tensor(arg)
                tensors.append(tensor)
                requires_grad = requires_grad or tensor.requires_grad

            if len(tensors) != n_in:
                raise ValueError(f"Number of input tensors does not match.")

            new_args = (*tensors, *args[n_in:])
            result_grad_func_pairs = func(*new_args, **kwargs)

            if n_ret == 1:
                result_grad_func_pairs = (result_grad_func_pairs,)

            if len(result_grad_func_pairs) != n_ret:
                raise ValueError(f"Number of returned tensors does not match.")

            results = []
            for result, compute_grad in result_grad_func_pairs:
                result.requires_grad = requires_grad and has_gradient
                results.append(result)

                def _backward_op(_func: callable = compute_grad) -> None:
                    grads = _func()
                    for tensor, grad in zip(tensors, grads):
                        new_grad = _match_grad_shape(tensor.data, grad)
                        _set_tensor_grad(tensor, new_grad)

                if not lucid.grad_enabled():
                    continue

                result._backward_op = _backward_op
                result._prev = tensors

            return tuple(results) if n_ret > 1 else results[0]

        return wrapper

    return decorator


def create_bfunc_op(has_gradient: bool = True) -> callable:
    return create_func_op(n_in=2, n_ret=1, has_gradient=has_gradient)


def create_ufunc_op(has_gradient: bool = True) -> callable:
    return create_func_op(n_in=1, n_ret=1, has_gradient=has_gradient)
