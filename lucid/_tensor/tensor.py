from typing import Callable, Iterator, Optional, Self, SupportsIndex
import numpy as np

import lucid
from lucid.types import _ArrayOrScalar, _NumPyArray, _Scalar, _DeviceType

from lucid._tensor.tensor_ops import _TensorOps
from lucid._backend.metal import mx, _MLXArray


_HookType = Callable[["Tensor", _NumPyArray | _MLXArray], None]


class Tensor(_TensorOps):
    def __init__(
        self,
        data: _ArrayOrScalar,
        requires_grad: bool = False,
        keep_grad: bool = False,
        dtype: type | None = None,
        device: _DeviceType = "cpu",
    ) -> None:
        if not isinstance(data, (_NumPyArray, _MLXArray)):
            self.data = np.array(data, dtype=dtype)
            self.dtype = self.data.dtype
        else:
            if dtype is not None and data.dtype != dtype:
                data = data.astype(dtype)
            self.data = data

        if device not in {"cpu", "gpu"}:
            raise ValueError(
                f"Unknown device type '{device}'. Must be either 'cpu' or 'gpu'."
            )
        if device == "gpu":
            self.data = mx.array(self.data)

        self.requires_grad = requires_grad and lucid.grad_enabled()
        self.keep_grad = keep_grad
        self.dtype = self.data.dtype
        self.device = device

        self.grad: Optional[_NumPyArray | _MLXArray] = None

        self._op: type | None = None
        self._backward_op: Callable = lambda: None
        self._prev: list[Tensor] = []
        self._backward_hooks: list[_HookType] = []

    @property
    def is_leaf(self) -> bool:
        return self.requires_grad and len(self._prev) == 0

    def backward(self, keep_grad: bool = False) -> None:
        if self.grad is None:
            if self.is_cpu():
                self.grad = np.ones_like(self.data)
            else:
                self.grad = mx.ones_like(self.data)

        visited = set()
        topo_order: list[Self] = []
        stack = [self]

        while stack:
            tensor = stack.pop()
            if tensor in visited:
                topo_order.append(tensor)
                continue

            visited.add(tensor)
            stack.append(tensor)
            for parent in tensor._prev:
                if parent not in visited:
                    stack.append(parent)

        for tensor in reversed(topo_order):
            try:
                tensor._backward_op()
            except Exception as e:
                raise RuntimeError(
                    f"Exception above occurred for tensor "
                    + f"of shape {tensor.shape} on operation {self._op}."
                ) from e

            for hook in tensor._backward_hooks:
                hook(tensor, tensor.grad)

            if not (tensor.is_leaf or keep_grad or self.keep_grad):
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

    def item(self) -> _Scalar:
        if self.ndim != 0:
            raise ValueError(
                "Tensor must be 0-dimensional(scalar) to pop its item.",
            )
        item = self.data
        if item % 1 == 0:
            return int(item)
        else:
            return float(item)

    def zero(self) -> None:
        if self.is_cpu():
            self.data = np.zeros_like(self.data)
        else:
            self.data = mx.zeros_like(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    def astype(self, dtype: type) -> Self:  # TODO: Need to modify this.
        self.data = self.data.astype(dtype)
        self.dtype = self.data.dtype
        return self

    def to(self, device: _DeviceType) -> Self:
        if self.device == device:
            return self

        if device == "cpu":
            self.data = np.array(self.data)
            if self.grad is not None:
                self.grad = np.array(self.grad)

        elif device == "gpu":
            self.data = mx.array(self.data)
            if self.grad is not None:
                self.grad = mx.array(self.grad)

        else:
            raise ValueError(
                f"Unknown device type '{device}'. Must be either 'cpu' or 'gpu'."
            )

        self.dtype = self.data.dtype
        self.device = device
        return self

    def is_cpu(self) -> bool:
        return self.device == "cpu"

    def is_gpu(self) -> bool:
        return self.device == "gpu"

    def __getitem__(self, idx: SupportsIndex) -> Self:
        new_idx = idx
        if isinstance(idx, Tensor):
            new_idx = idx.data
        if isinstance(idx, tuple):
            new_idx = tuple()
            for id in idx:
                if isinstance(id, Tensor):
                    id = id.data
                new_idx += (id,)

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

        if self.requires_grad:
            new_tensor._backward_op = _backward_op
            new_tensor._prev = [self]

        return new_tensor

    def __setitem__(self, idx: SupportsIndex, value: Self | _ArrayOrScalar) -> None:
        if self.requires_grad:
            raise RuntimeError(
                "Cannot perform in-place item setting on a "
                + "Tensor that requires gradients. "
            )

        if not isinstance(value, Tensor):
            value = Tensor(value)
        self.data[idx] = value.data

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
