from typing import Any, Optional, Self
import numpy as np


_Scalar = int | float
_NumPyArray = np.ndarray
_ArrayOrScalar = _Scalar | list | _NumPyArray


class Tensor:
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

    def __hash__(self) -> int:
        return hash(id(self))

    def __add__(self, _: Self) -> Self: ...

    def __radd__(self, _: Self) -> Self: ...

    def __sub__(self, _: Self) -> Self: ...

    def __rsub__(self, _: Self) -> Self: ...

    def __mul__(self, _: Self) -> Self: ...

    def __rmul__(self, _: Self) -> Self: ...

    def __truediv__(self, _: Self) -> Self: ...

    def __rtrudiv__(self, _: Self) -> Self: ...

    def __eq__(self, _: Self) -> Self: ...

    def __gt__(self, _: Self) -> Self: ...

    def __lt__(self, _: Self) -> Self: ...

    def __pow__(self, _: _Scalar) -> Self: ...

    def __neg__(self) -> Self: ...
