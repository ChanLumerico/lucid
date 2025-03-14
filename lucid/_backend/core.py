import functools
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Any

import lucid
from lucid.types import _DeviceType, _NumPyArray
from lucid._tensor import Tensor
from lucid._backend.metal import _MLXArray, _is_cpu_op


_GradFuncType = Callable[
    [None], _NumPyArray | _MLXArray | tuple[_NumPyArray | _MLXArray, ...]
]

_ReturnGradFuncPair = Tuple[Tensor, _GradFuncType]

_FuncOpReturnType = _ReturnGradFuncPair | Tuple[_ReturnGradFuncPair, ...]


def func_op(
    n_in: int | None,
    n_ret: int | None,
    has_gradient: bool = True,
    device: _DeviceType = "cpu",
) -> Callable:

    def decorator(func: Callable[..., _FuncOpReturnType]) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Tuple[Tensor, ...]:
            tensors: Tuple[Tensor, ...] = tuple()
            requires_grad = False

            op_self = args[0]
            args = args[1:]
            if n_in is None:
                tensor_args = args
            else:
                if len(args) < n_in:
                    raise ValueError(
                        f"Expected at least {n_in} tensor arguments, got {len(args)}"
                    )
                tensor_args = args[:n_in]

            for arg in tensor_args:
                tensor = lucid._check_is_tensor(arg, device=device)
                tensors += (tensor,)
                requires_grad = requires_grad or tensor.requires_grad

            non_tensor_args = args[n_in:] if n_in is not None else ()
            new_args = (*tensors, *non_tensor_args)

            func_return_pairs = func(op_self, *new_args, **kwargs)

            if n_ret is None:
                if not isinstance(func_return_pairs, tuple):
                    raise ValueError(
                        f"{func.__name__} should return multiple '_ReturnGradFuncPair'."
                    )
                num_returns = len(func_return_pairs)
            else:
                num_returns = n_ret

            if num_returns == 1:
                func_return_pairs = (func_return_pairs,)

            results: Tuple[Tensor, ...] = tuple()
            for result, compute_grad in func_return_pairs:
                result.requires_grad = requires_grad and has_gradient
                result.to(device)
                result._op = type(op_self)

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
                        new_grad = lucid._match_grad_shape(
                            tensor.data, grad, device=device
                        )
                        lucid._set_tensor_grad(tensor, new_grad)

                if not lucid.grad_enabled():
                    continue

                if result.requires_grad:
                    result._backward_op = _backward_op
                    result._prev = list(tensors)

            return results if num_returns > 1 else results[0]

        return wrapper

    return decorator


def unary_func_op(has_gradient: bool = True, device: _DeviceType = "cpu") -> Callable:
    return func_op(1, 1, has_gradient=has_gradient, device=device)


def binary_func_op(has_gradient: bool = True, device: _DeviceType = "cpu") -> Callable:
    return func_op(2, 1, has_gradient=has_gradient, device=device)


def poly_func_op(has_gradient: bool = True, device: _DeviceType = "cpu") -> Callable:
    return func_op(None, 1, has_gradient=has_gradient, device=device)


class operation(ABC):
    def __init__(self) -> None:
        self.result: Tensor | None = None

    @abstractmethod
    def cpu(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]: ...

    @abstractmethod
    def gpu(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]: ...

    def compute_grad(self, *args, **kwargs) -> _GradFuncType: ...

    def compute_grad_cpu(self, *args, **kwargs) -> _GradFuncType: ...

    def compute_grad_gpu(self, *args, **kwargs) -> _GradFuncType: ...

    def __call__(self, *tensors, **kwargs) -> Tensor | tuple[Tensor, ...]:
        return (
            self.cpu(*tensors, **kwargs)
            if _is_cpu_op(*tensors)
            else self.gpu(*tensors, **kwargs)
        )
