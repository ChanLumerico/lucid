from abc import ABC, abstractmethod
from typing import Callable, Self, Tuple, ClassVar
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
            inplace_target: _TensorLike | None = None

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

            if op_self._inplace:
                if n_ret != 1:
                    raise ValueError("inplace op must have a single return value.")
                if not (0 <= op_self._inplace_target < len(tensors)):
                    raise ValueError("inplace_target is out of range.")

                target = tensors[op_self._inplace_target]
                if lucid.grad_enabled() and target.requires_grad and target.is_leaf:
                    raise RuntimeError(
                        "A leaf tensor with 'requires_grad=True' "
                        "cannot be subjected to inplace operations."
                    )
                inplace_target = target

                proxy = target.new_tensor()
                proxy._op = target._op
                proxy._prev = list(target._prev)
                proxy._backward_op = target._backward_op
                proxy._backward_hooks = list(target._backward_hooks)
                proxy.grad = None
                proxy._version = target._version

                if hasattr(target, "_is_free"):
                    proxy._is_free = target._is_free
                if hasattr(target, "_is_bool_tensor"):
                    proxy._is_bool_tensor = target._is_bool_tensor

                tensors = (
                    tensors[: op_self._inplace_target]
                    + (proxy,)
                    + tensors[op_self._inplace_target + 1 :]
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
            elif op_self._inplace:
                raise ValueError("inplace op must have a single return value.")

            if op_self._inplace:
                (ret_value, grad_func) = func_return_pairs[0]
                target = inplace_target
                if target is None:
                    raise RuntimeError("Missing inplace target tensor.")
                target.data = ret_value.data

                if ret_value.dtype is bool:
                    target._is_bool_tensor = True
                    target.dtype = bool
                else:
                    target._is_bool_tensor = False
                    target.dtype = ret_value.dtype

                target._version += 1
                func_return_pairs = ((target, grad_func),)

            results: Tuple[_TensorLike, ...] = tuple()
            for result, grad_func in func_return_pairs:
                result.requires_grad = requires_grad and has_gradient and grad_enabled
                result.to(device)
                result.free() if is_free else ...
                results += (result,)

                if not track_graph:
                    continue
                result._op = op_self

                if result.requires_grad or lucid.flops_enabled():
                    result._prev = list(tensors)
                    if not result.requires_grad:
                        continue

                    result._backward_op = BackwardOperation(
                        forward_op_ref=weakref.ref(op_self),
                        grad_func=grad_func,
                        tensor_refs=tensor_refs,
                        versions=tuple(t._version for t in tensors),
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
    return func_op(1, 1, has_gradient, device)


def binary_func_op(has_gradient: bool = True, device: _DeviceType = "cpu") -> Callable:
    return func_op(2, 1, has_gradient, device)


def poly_func_op(has_gradient: bool = True, device: _DeviceType = "cpu") -> Callable:
    return func_op(None, 1, has_gradient, device)


class Operation(ABC):
    __fallback__: ClassVar[bool] = False

    def __init__(self) -> None:
        self.result: _TensorLike | tuple[_TensorLike, ...] | None = None
        self._inplace: bool = False
        self._inplace_target: int = 0
        self._flops: int | None = None

    def clear(self) -> None:
        self.result = None

    def inplace(self, target: int = 0) -> Self:
        self._inplace = True
        self._inplace_target = target
        return self

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
        versions: tuple[int, ...] = (),
        device: _DeviceType | None = "cpu",
        custom_closure: Callable[[], None] | None = None,
    ) -> None:
        self.forward_op_ref = forward_op_ref
        self.grad_func = grad_func
        self.tensor_refs = tensor_refs
        self.versions = versions
        self.device = device

        self.custom_closure = custom_closure
        self.num_inputs = len(tensor_refs)

        if self.grad_func is None and self.custom_closure is None:
            raise ValueError("Either 'grad_func' or 'custom_closure' must be provided.")

        if len(tensor_refs) != len(versions):
            raise ValueError("Numbers of 'tensor_refs' and 'versions' do not match.")

    def override_grad_func(self, new_grad_func: _GradFuncType) -> None:
        if self.custom_closure is not None:
            return
        self.grad_func = new_grad_func

    def override_tensor_refs(
        self,
        new_tensor_refs: tuple[weakref.ref[_TensorLike]],
        new_versions: tuple[int, ...] | None = None,
    ) -> None:
        self.tensor_refs = new_tensor_refs
        self.num_inputs = len(new_tensor_refs)
        if new_versions is not None:
            if len(new_versions) != len(new_tensor_refs):
                raise ValueError(
                    "Numbers of 'tensor_refs' and 'versions' do not match."
                )
            self.versions = new_versions

    def __call__(self) -> None:
        live_tensors = tuple(ref() for ref in self.tensor_refs)
        if not live_tensors or any(t is None for t in live_tensors):
            return

        if any(ver != t._version for ver, t in zip(self.versions, live_tensors)):
            raise RuntimeError(f"Tensor version mismatch detected.")

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

        if len(grads) != len(live_tensors):
            raise ValueError(
                f"Expected {len(live_tensors)} gradients, got {len(grads)}."
            )

        for tensor, grad in zip(live_tensors, grads):
            if not tensor.requires_grad and grad is None:
                continue
            new_grad = lucid._match_grad_shape(tensor.data, grad, device=self.device)
            lucid._set_tensor_grad(tensor, new_grad)


noop = BackwardOperation(
    forward_op_ref=None, grad_func=lambda: (), tensor_refs=(), device=None
)
