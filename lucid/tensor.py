from typing import Optional, Self
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

    def __add__(self, other: Self) -> Self:
        return _add(self, other)

    def __sub__(self, other: Self) -> Self:
        return _sub(self, other)

    def __mul__(self, other: Self) -> Self:
        return _mul(self, other)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __str__(self) -> str:
        return self.data.__str__()


# [binary] addition
def _add(self: Tensor, other: Tensor) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)

    result = Tensor(
        self.data + other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def _backward_add() -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.copy(result.grad)
            else:
                self.grad += result.grad

        if other.requires_grad:
            if other.grad is None:
                other.grad = np.copy(result.grad)
            else:
                other.grad += result.grad

    result._backward_op = _backward_add
    result._prev = [self, other]
    return result


# [binary] subtraction
def _sub(self: Tensor, other: Tensor) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)

    result = Tensor(
        self.data - other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def _backward_sub() -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.copy(result.grad)
            else:
                self.grad += result.grad

        if other.requires_grad:
            if other.grad is None:
                other.grad = -np.copy(result.grad)
            else:
                other.grad -= result.grad

    result._backward_op = _backward_sub
    result._prev = [self, other]
    return result


# [binary] element-wise multiplication
def _mul(self: Tensor, other: Tensor) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)

    result = Tensor(
        self.data * other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def _backward_mul() -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = other.data * result.grad
            else:
                self.grad += other.data * result.grad

        if other.requires_grad:
            if other.grad is None:
                other.grad = self.data * result.grad
            else:
                other.grad -= self.data * result.grad

    result._backward_op = _backward_mul
    result._prev = [self, other]
    return result


a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

c = a * b + a
c.backward()

print(a.grad, b.grad, c.grad)
