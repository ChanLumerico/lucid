from abc import ABC, abstractmethod
from typing import Callable, Tuple, ClassVar
import functools

import lucid
import lucid.types as types
from lucid.types import _DeviceType, _NumPyArray, _MLXArray

from lucid._tensor import Tensor
from lucid._backend.metal import is_gpu_op


_GradFuncType = Callable[[None], Tuple[_NumPyArray | _MLXArray, ...]]

_ReturnGradFuncPair = Tuple[Tensor, _GradFuncType]

_FuncOpReturnType = Tuple[_ReturnGradFuncPair, ...]


def func_op(
    n_in: int | None,
    n_ret: int | None,
    has_gradient: bool = True,
    device: _DeviceType = "cpu",
) -> Callable:
    def decorator(func: Callable[..., _FuncOpReturnType]) -> Callable:
        @functools.wraps(func)
        def wrapper(op_self: operation, *args, **kwargs) -> Tuple[Tensor, ...]:
            tensors: Tuple[Tensor, ...] = tuple()
            requires_grad = False
            is_free = True

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

                if tensor.is_free:
                    tensor.to(device)
                else:
                    is_free = False
                    if tensor.device != device:
                        raise RuntimeError(
                            f"{tensor.device} tensor of shape {tensor.shape} "
                            + f"passed for {device} operation"
                            + f"('{type(op_self).__name__}')."
                        )

            non_tensor_args = args[n_in:] if n_in is not None else ()
            new_args = (*tensors, *non_tensor_args)
            func_return_pairs = func(op_self, *new_args, **kwargs)

            flops_prev, flops_new = 0, 0
            if lucid.flops_enabled():
                flops_prev = sum(t.flops for t in tensors)
                flops_new = flops_prev + op_self.flops

            if n_ret is None:
                if not isinstance(func_return_pairs, tuple):
                    raise ValueError(
                        f"{func.__name__} should return multiple '_ReturnGradFuncPair'."
                    )
                num_returns = len(func_return_pairs)
            else:
                num_returns = n_ret

            if num_returns == 1:
                func_return_pairs: _FuncOpReturnType = (func_return_pairs,)

            results: Tuple[Tensor, ...] = tuple()
            for result, compute_grad in func_return_pairs:
                result.requires_grad = requires_grad and has_gradient

                result.to(device)
                result.dtype = types.to_numeric_type(result.data.dtype)

                result._op = type(op_self)
                result._flops += flops_new
                if is_free:
                    result.free()

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
    __fallback__: ClassVar[bool] = False

    def __init__(self) -> None:
        self.result: Tensor | tuple[Tensor, ...] | None = None
        self._flops: int = 0

    @abstractmethod
    def cpu(self, *args, **kwargs) -> _FuncOpReturnType: ...

    @abstractmethod
    def gpu(self, *args, **kwargs) -> _FuncOpReturnType: ...

    def __grad__(self, *args, **kwargs) -> _GradFuncType: ...

    def __grad_cpu__(self, *args, **kwargs) -> _GradFuncType: ...

    def __grad_gpu__(self, *args, **kwargs) -> _GradFuncType: ...

    @abstractmethod
    def __flops__(self, *args, **kwargs) -> int: ...

    @property
    def flops(self) -> int | None:
        return self._flops

    def __call__(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]:
        if lucid.flops_enabled():
            self._flops += self.__flops__(*args, **kwargs)

        if is_gpu_op(*args):
            return self.gpu(*args, **kwargs)
        return self.cpu(*args, **kwargs)


def fallback(cls: type[operation]) -> type[operation]:
    cls.__fallback__ = True
    return cls
