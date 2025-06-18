from typing import Callable, Iterator, Optional, Self, SupportsIndex, Any
from types import NoneType
from collections import deque
import numpy as np

import lucid
import lucid.types as types
from lucid.types import (
    _ArrayOrScalar,
    _NumPyArray,
    _MLXArray,
    _Scalar,
    _DeviceType,
    _BuiltinNumeric,
    Numeric,
)

from lucid._tensor.tensor_ops import _TensorOps
from lucid._backend.metal import mx, parse_mlx_indexing, check_metal_availability


_HookType = Callable[["Tensor", _NumPyArray | _MLXArray], None]

_dtype_map = {int: types.Int64, float: types.Float64, complex: types.Complex64}


class Tensor(_TensorOps):
    def __init__(
        self,
        data: _ArrayOrScalar | _MLXArray,
        requires_grad: bool = False,
        keep_grad: bool = False,
        dtype: _BuiltinNumeric | Numeric | None = None,
        device: _DeviceType = "cpu",
    ) -> None:
        self._is_free = False
        self._is_bool_tensor = False

        if dtype is bool:
            self._is_bool_tensor = True
            dtype = None
        if dtype in {int, float, complex}:
            dtype = _dtype_map[dtype]

        dtype: Numeric | None
        assert isinstance(dtype, (Numeric, NoneType)), type(dtype)

        if dtype is not None and dtype.is_bit_free:
            raise TypeError(
                f"Implicit dtypes is not supported for Tensor instantiation."
            )
        if device not in {"cpu", "gpu"}:
            raise lucid.UnknownDeviceError(device)

        self.data: _NumPyArray | _MLXArray
        if not isinstance(data, (_NumPyArray, _MLXArray)):
            self.data = np.array(data, dtype=dtype.cpu if dtype is not None else dtype)

            if isinstance(data, (_Scalar, list, tuple)):
                self._is_free = True

        elif isinstance(data, _NumPyArray):
            self.data = data
            if dtype is not None and data.dtype != dtype.cpu:
                self.data = data.astype(dtype.cpu)

        elif isinstance(data, _MLXArray):
            check_metal_availability()
            device = "gpu"
            self.data = data
            if dtype is not None and data.dtype != dtype.gpu:
                self.data = data.astype(dtype.gpu)

        if device == "gpu" and isinstance(self.data, _NumPyArray):
            check_metal_availability()
            self.data = mx.array(self.data)

        if "bool" in str(self.data.dtype):
            self._is_bool_tensor = True
        else:
            if self._is_bool_tensor:
                self.data = self.data.astype(bool if device == "cpu" else mx.bool_)

        self._op: object | None = None
        self._backward_op: Callable = lambda: None
        self._prev: list[Tensor] = []
        self._backward_hooks: list[_HookType] = []

        self.grad: Optional[_NumPyArray | _MLXArray] = None

        self.device = device
        self.requires_grad = requires_grad and lucid.grad_enabled()
        self.keep_grad = keep_grad

        self.dtype: Numeric | bool
        if self._is_bool_tensor:
            self.dtype = bool
        else:
            self.dtype = (
                dtype if dtype is not None else types.to_numeric_type(self.data.dtype)
            )

    @property
    def is_leaf(self) -> bool:
        return self.requires_grad and len(self._prev) == 0

    @property
    def is_free(self) -> bool:
        return self._is_free

    @classmethod
    def is_all_free(cls, *tensors: Self) -> bool:
        return all(t.is_free for t in tensors)

    @classmethod
    def copy_data(cls, data: _NumPyArray | _MLXArray) -> _NumPyArray | _MLXArray:
        if isinstance(data, _NumPyArray):
            return data.copy()
        elif isinstance(data, _MLXArray):
            return mx.array(data)
        else:
            raise TypeError(f"Unexpected type: '{type(data).__name__}'")

    @classmethod
    def copy_grad(cls, grad: _NumPyArray | _MLXArray) -> _NumPyArray | _MLXArray:
        return cls.copy_data(data=grad)

    def free(self) -> Self:
        self._is_free = True
        return self

    def eval(self) -> Self:
        if self.is_gpu():
            mx.eval(self.data)
            mx.stop_gradient(self.data)
        return self

    def backward(self, keep_grad: bool = False) -> None:
        if self.grad is None:
            self.grad = (
                np.ones_like(self.data) if self.is_cpu() else mx.ones_like(self.data)
            )
        visited = set()
        topo_order: list[Self] = []
        stack = [self]

        while stack:
            tensor = stack[-1]
            if tensor in visited:
                stack.pop()
                topo_order.append(tensor)
                continue

            visited.add(tensor)
            for parent in tensor._prev:
                if parent not in visited:
                    stack.append(parent)

        for tensor in reversed(topo_order):
            try:
                tensor._backward_op()
            except Exception as e:
                raise lucid.BackwardError(shape=tensor.shape, op=self._op) from e

            for hook in tensor._backward_hooks:
                hook(tensor, tensor.grad)

            if not (tensor.is_leaf or keep_grad or tensor.keep_grad):
                tensor.grad = None

    def register_hook(self, hook: _HookType) -> Callable:
        self._backward_hooks.append(hook)
        return lambda: self._backward_hooks.remove(hook)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def flops(self) -> int:
        total = 0
        queue = deque([self])
        visited = set()

        while queue:
            cur = queue.popleft()
            visited.add(cur)
            if cur._op is not None:
                total += cur._op.flops

            for next in cur._prev:
                if next not in visited:
                    queue.appendleft(next)

        return total

    def item(self) -> _Scalar:
        if self.size > 1:
            raise ValueError(
                "Tensor must be 0-dimensional(scalar) to pop its item. "
                "Use `tensor.data` to access the data directly.",
            )
        item = self.data[..., 0] if self.ndim > 0 else self.data
        if item % 1 == 0:
            return int(item)
        else:
            return float(item)

    def tolist(self) -> list:
        return self.data.tolist()

    def numpy(self) -> _NumPyArray:
        return np.array(self.data)

    def mlx(self) -> _MLXArray:
        return mx.array(self.data)

    def zero(self) -> None:
        if self.is_cpu():
            self.data = np.zeros_like(self.data)
        else:
            self.data = mx.zeros_like(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    def astype(self, dtype: type | Numeric) -> Self:
        new_dtype = dtype
        if isinstance(dtype, Numeric):
            self._is_bool_tensor = False
            if dtype.is_bit_free:
                new_dtype = dtype.auto_parse(self.data.dtype, device=self.device)
            else:
                new_dtype = dtype.parse(device=self.device)

        elif dtype is bool:
            self._is_bool_tensor = True
            if self.is_gpu():
                new_dtype = mx.bool_

        self.data = self.data.astype(new_dtype)
        if self._is_bool_tensor:
            self.dtype = bool
        else:
            self.dtype = types.to_numeric_type(self.data.dtype)

        return self

    def to(self, device: _DeviceType) -> Self:
        if self.device == device:
            return self

        if device == "cpu":
            self.data = np.array(self.data)
            if self.grad is not None:
                self.grad = np.array(self.grad)

        elif device == "gpu":
            check_metal_availability()
            self.data = mx.array(self.data)
            if self.grad is not None:
                self.grad = mx.array(self.grad)

        else:
            raise lucid.UnknownDeviceError(device)

        self.device = device
        return self

    def is_cpu(self) -> bool:
        return self.device == "cpu"

    def is_gpu(self) -> bool:
        return self.device == "gpu"

    def __getitem__(self, idx: SupportsIndex | Self) -> Self:
        new_idx = idx
        if isinstance(idx, Tensor):
            new_idx = idx.data
            if self.is_gpu() and idx.is_free:
                new_idx = mx.array(idx.data)

        if isinstance(idx, tuple):
            new_idx = tuple()
            for id in idx:
                if isinstance(id, Tensor):
                    if self.is_gpu() and id.is_free:
                        id = mx.array(id.data)
                    else:
                        id = id.data
                new_idx += (id,)

        if self.is_gpu():
            new_idx = parse_mlx_indexing(new_idx)
        else:
            if isinstance(idx, Tensor) and idx.is_gpu():
                raise lucid.DeviceMismatchError(to="gpu", from_="cpu")

        sliced_data = self.data[new_idx]
        new_tensor = Tensor(
            sliced_data, self.requires_grad, self.keep_grad, self.dtype, self.device
        )

        def _backward_op() -> None:
            if self.grad is None:
                self.grad = (
                    np.zeros_like(self.data)
                    if self.is_cpu()
                    else mx.zeros_like(self.data)
                )

            new_grad = lucid._match_grad_shape(
                self.data[new_idx], new_tensor.grad, device=self.device
            )
            lucid._set_tensor_grad(self, new_grad, at=new_idx)

        if self.requires_grad and lucid.grad_enabled():
            new_tensor._backward_op = _backward_op
            new_tensor._prev = [self]

        if self.is_free:
            new_tensor.free()

        return new_tensor

    def __setitem__(
        self, idx: SupportsIndex | Self, value: Self | _ArrayOrScalar
    ) -> None:
        if self.requires_grad:
            raise RuntimeError(
                "Cannot perform in-place item setting on a "
                + "Tensor that requires gradients. "
            )

        new_idx = idx
        if isinstance(idx, Tensor):
            new_idx = idx.data
        elif isinstance(idx, tuple):
            new_idx = tuple()
            for id in idx:
                if isinstance(id, Tensor):
                    id = id.data
                new_idx += (id,)

        if self.is_gpu():
            new_idx = parse_mlx_indexing(new_idx)

        data: _ArrayOrScalar
        if isinstance(value, Tensor):
            if self.device != value.device:
                if value.is_free:
                    data = value.to(self.device).data
                else:
                    raise lucid.DeviceMismatchError(to=self.device, from_=value.device)
            data = value.data
        else:
            data = value
        self.data[new_idx] = data

    def __len__(self) -> int:
        if self.ndim == 0:
            return self.size
        else:
            return self.shape[0]

    def __iter__(self) -> Iterator[Self]:
        for i in range(self.shape[0]):
            yield self[i]

    def __repr__(self) -> str:
        return f"Tensor({self.data}, grad={self.grad}, device={self.device})"

    def __str__(self) -> str:
        return str(self.data)

    def __hash__(self) -> int:
        return hash(id(self))

    def __deepcopy__(self, *args: Any) -> Self:
        copied_data = Tensor.copy_data(self.data)

        new_tensor = Tensor(
            copied_data,
            requires_grad=self.requires_grad,
            keep_grad=self.keep_grad,
            dtype=self.dtype,
            device=self.device,
        )

        if self.grad is not None:
            new_tensor.grad = Tensor.copy_grad(self.grad)

        new_tensor._op = self._op
        new_tensor._backward_op = self._backward_op
        new_tensor._prev = self._prev.copy()
        new_tensor._backward_hooks = self._backward_hooks.copy()

        new_tensor._is_free = self._is_free
        new_tensor._is_bool_tensor = self._is_bool_tensor

        return new_tensor

    def __bool__(self) -> bool:
        if self.data.size != 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one value is ambiguous. "
                "Use `tensor.any()` or `tensor.all()` instead."
            )
        return bool(self.data.item())

    def any(self) -> bool:
        if self.is_cpu():
            return bool(np.any(self.data))
        else:
            mx.eval(self.data)
            return bool(mx.any(self.data).item())

    def all(self) -> bool:
        if self.is_cpu():
            return bool(np.all(self.data))
        else:
            mx.eval(self.data)
            return bool(mx.all(self.data).item())
