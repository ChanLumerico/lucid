from typing import Any, Iterator, Optional, Self, SupportsIndex
import numpy as np
import lucid
from lucid._tensor.tensor_ops import _TensorOps
from lucid.types import _ArrayOrScalar, _NumPyArray, _Scalar


class Tensor(_TensorOps):
    def __init__(
        self,
        data: _ArrayOrScalar,
        requires_grad: bool = False,
        keep_grad: bool = False,
        dtype: Any = np.float32,
    ) -> None:
        if not isinstance(data, _NumPyArray):
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = data

        self.requires_grad = requires_grad and lucid.grad_enabled()
        self.keep_grad = keep_grad
        self.dtype = self.data.dtype

        self.grad: Optional[_NumPyArray] = None
        self._backward_op: callable = lambda: None
        self._prev: list[Tensor] = []

    @property
    def is_leaf(self) -> bool:
        return self.requires_grad and len(self._prev) == 0

    def backward(self, keep_grad: bool = False) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        visited = set()
        topo_order: list[Tensor] = []

        def _build_topo(tensor: Tensor) -> None:
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    _build_topo(parent)
                topo_order.append(tensor)

        _build_topo(self)
        topo_order.reverse()

        for tensor in topo_order:
            tensor._backward_op()
            if not tensor.is_leaf and not keep_grad:
                self.zero_grad()

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
        self.data = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        if not self.keep_grad:
            self.grad = None

    def __getitem__(self, idx: SupportsIndex) -> Self:
        sliced_data = self.data[idx]
        new_tensor = Tensor(sliced_data, requires_grad=self.requires_grad)

        def _backward_op() -> None:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)

            new_grad = lucid._match_grad_shape(self.data[idx], new_tensor.grad)
            lucid._set_tensor_grad(self, new_grad, at=idx)

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

    def __iter__(self) -> Iterator[Self]:
        for i in range(self.shape[0]):
            yield self[i]

    def __repr__(self) -> str:
        return f"Tensor({self.data}, grad={self.grad})"

    def __str__(self) -> str:
        return str(self.data)

    def __hash__(self) -> int:
        return hash(id(self))
