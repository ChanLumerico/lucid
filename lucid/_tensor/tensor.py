from typing import Any, Optional
import numpy as np

from lucid._tensor.tensor_ops import _TensorOps
from lucid.types import _ArrayOrScalar, _NumPyArray


class Tensor(_TensorOps):
    def __init__(
        self,
        data: _ArrayOrScalar,
        requires_grad: bool = False,
        dtype: Any = np.float32,
    ) -> None:
        if not isinstance(data, _NumPyArray):
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = data

        self.requires_grad = requires_grad
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
                tensor.grad = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __str__(self) -> str:
        return str(self.data)

    def __hash__(self) -> int:
        return hash(id(self))
