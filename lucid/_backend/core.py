from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, ClassVar
import functools
import weakref

import lucid
from lucid.types import (
    Numeric,
    _DeviceType,
    _NumPyArray,
    _MLXArray,
    _BuiltinNumeric,
    _TensorLike,
)

from lucid._backend.metal import is_gpu_op


_GradType = _NumPyArray | _MLXArray | Tuple[_NumPyArray | _MLXArray, ...]
_GradFuncType = Callable[[], _GradType]


_ReturnGradFuncPair = Tuple[_TensorLike, _GradFuncType]
_FuncOpReturnType = _ReturnGradFuncPair | Tuple[_ReturnGradFuncPair, ...]


def func_op(
    n_in: int | None,
    n_ret: int | None,
    has_gradient: bool = True,
    device: _DeviceType = "cpu",
) -> Callable:
    def decorator(forward_func: Callable[..., _FuncOpReturnType]) -> Callable:
        @functools.wraps(forward_func)
        def wrapper(op_self: Operation, *args, **kwargs) -> Tuple[_TensorLike, ...]:
            tensors: Tuple[_TensorLike, ...] = tuple()
            requires_grad = False
            is_free = True
            dtype_hint: _BuiltinNumeric | Numeric | None = None

            if n_in is None:
                tensor_args = args
            else:
                if len(args) < n_in:
                    raise ValueError(
                        f"Expected at least {n_in} tensor arguments, got {len(args)}"
                    )
                tensor_args = args[:n_in]

            for arg in tensor_args:
                if isinstance(arg, _TensorLike):
                    dtype_hint = arg.dtype
                    break

            for arg in tensor_args:
                tensor = lucid._check_is_tensor(arg, device=device, dtype=dtype_hint)
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
            func_return_pairs = forward_func(op_self, *new_args, **kwargs)

            tensor_refs = tuple(weakref.ref(t) for t in tensors)

            grad_enabled = lucid.grad_enabled()
            flops_enabled = lucid.flops_enabled()
            track_graph = flops_enabled or (grad_enabled and requires_grad)

            if flops_enabled:
                op_self.flops = op_self.__flops__(*new_args, **kwargs)

            if n_ret is None:
                if not isinstance(func_return_pairs, tuple):
                    raise ValueError(
                        f"{forward_func.__name__} should return multiple '_ReturnGradFuncPair'."
                    )
                num_returns = len(func_return_pairs)
            else:
                num_returns = n_ret

            if num_returns == 1:
                func_return_pairs: _FuncOpReturnType = (func_return_pairs,)

            results: Tuple[_TensorLike, ...] = tuple()
            for result, grad_func in func_return_pairs:
                result.requires_grad = requires_grad and has_gradient and grad_enabled
                if track_graph:
                    result._op = op_self
                result.to(device)
                if is_free:
                    result.free()

                results += (result,)
                if not track_graph:
                    continue

                if result.requires_grad or lucid.flops_enabled():
                    result._prev = list(tensors)
                    if not result.requires_grad:
                        continue
                    result._backward_op = BackwardOperation(
                        forward_op_ref=weakref.ref(op_self),
                        grad_func=grad_func,
                        tensor_refs=tensor_refs,
                        device=device,
                    )

            if track_graph:
                try:
                    op_self.result = results if num_returns > 1 else results[0]
                except Exception:
                    pass
            else:
                try:
                    op_self.clear()
                except Exception:
                    try:
                        op_self.result = None
                    except Exception:
                        pass

            return results if num_returns > 1 else results[0]

        return wrapper

    return decorator


def unary_func_op(has_gradient: bool = True, device: _DeviceType = "cpu") -> Callable:
    return func_op(1, 1, has_gradient=has_gradient, device=device)


def binary_func_op(has_gradient: bool = True, device: _DeviceType = "cpu") -> Callable:
    return func_op(2, 1, has_gradient=has_gradient, device=device)


def poly_func_op(has_gradient: bool = True, device: _DeviceType = "cpu") -> Callable:
    return func_op(None, 1, has_gradient=has_gradient, device=device)


class Operation(ABC):
    __fallback__: ClassVar[bool] = False

    def __init__(self) -> None:
        self.result: _TensorLike | tuple[_TensorLike, ...] | None = None
        self._flops: int | None = None

    def clear(self) -> None:
        self.result = None

    @abstractmethod
    def cpu(self, *args, **kwargs) -> _FuncOpReturnType: ...

    @abstractmethod
    def gpu(self, *args, **kwargs) -> _FuncOpReturnType: ...

    def __grad__(self, *args, **kwargs) -> _GradType: ...

    def __grad_cpu__(self, *args, **kwargs) -> _GradType: ...

    def __grad_gpu__(self, *args, **kwargs) -> _GradType: ...

    @property
    def flops(self) -> int:
        if self._flops is None:
            raise ValueError(f"flops counting for {self} has not been executed.")
        return self._flops

    @flops.setter
    def flops(self, val: int) -> None:
        self._flops = val

    def __flops__(self, *args, **kwargs) -> int:
        return 0

    def __call__(self, *args, **kwargs) -> _TensorLike | tuple[_TensorLike, ...]:
        if is_gpu_op(*args):
            return self.gpu(*args, **kwargs)
        return self.cpu(*args, **kwargs)


def fallback(cls: type[Operation]) -> type[Operation]:
    cls.__fallback__ = True
    return cls


class BackwardOperation:
    def __init__(
        self,
        forward_op_ref: weakref.ref[Operation] | None,
        grad_func: _GradFuncType | None,
        tensor_refs: tuple[weakref.ref[_TensorLike]],
        device: _DeviceType | None = "cpu",
        custom_closure: Callable[..., None] | None = None,
    ) -> None:
        self.forward_op_ref = forward_op_ref
        self.grad_func = grad_func
        self.tensor_refs = tensor_refs
        self.device = device

        self.custom_closure = custom_closure
        self.num_inputs = len(tensor_refs)

        if self.grad_func is None and self.custom_closure is None:
            raise ValueError("Either 'grad_func' or 'custom_closure' must be provided.")

    def override_grad_func(self, new_grad_func: _GradFuncType) -> None:
        if self.custom_closure is not None:
            return
        self.grad_func = new_grad_func

    def __call__(self) -> None:
        if self.custom_closure is not None:
            self.custom_closure()
            return

        if self.device is None and self.forward_op_ref is not None:
            raise RuntimeError(
                "Only 'noop' BackwardOperation can be called without device."
            )

        grads = self.grad_func()
        if self.num_inputs == 1 or not isinstance(grads, tuple):
            grads = (grads,)

        live_tensors = tuple(ref() for ref in self.tensor_refs)
        if any(t is None for t in live_tensors):
            return

        if len(grads) != len(live_tensors):
            raise ValueError(
                f"Expected {len(live_tensors)} gradients, got {len(grads)}."
            )

        for tensor, grad in zip(live_tensors, grads):
            new_grad = lucid._match_grad_shape(tensor.data, grad, device=self.device)
            lucid._set_tensor_grad(tensor, new_grad)


noop = BackwardOperation(
    forward_op_ref=None, grad_func=lambda: (), tensor_refs=(), device=None
)
