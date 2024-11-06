from typing import Optional
import numpy as np

_ArrayLike = list | np.ndarray


class Tensor:
    def __init__(self, data: _ArrayLike, requires_grad: bool = False) -> None:
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None

        self._backward_op: callable = lambda: None
        self._prev: list[Tensor] = []

    @property
    def is_leaf(self) -> bool:
        return self.requires_grad and len(self._prev) == 0

    def backward(self) -> None:
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
            if not tensor.is_leaf:
                tensor.grad = None

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __str__(self) -> str:
        return self.data.__str__()
