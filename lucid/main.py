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
        return _bfunc_add(self, other)

    def __sub__(self, other: Self) -> Self:
        return _bfunc_sub(self, other)

    def __mul__(self, other: Self) -> Self:
        return _bfunc_mul(self, other)

    def __truediv__(self, other: Self) -> Self:
        return _bfunc_truediv(self, other)

    def __pow__(self, exp: float) -> Self:
        return _ufunc_pow(self, exp)

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __str__(self) -> str:
        return self.data.__str__()


def _set_tensor_grad(tensor: Tensor, grad: np.ndarray) -> None:
    if tensor.requires_grad:
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad += grad


# [binary] addition
def _bfunc_add(self: Tensor, other: Tensor) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)

    result = Tensor(
        self.data + other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def _backward() -> None:
        _set_tensor_grad(self, result.grad)
        _set_tensor_grad(other, result.grad)

    result._backward_op = _backward
    result._prev = [self, other]
    return result


# [binary] subtraction
def _bfunc_sub(self: Tensor, other: Tensor) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)

    result = Tensor(
        self.data - other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def _backward() -> None:
        _set_tensor_grad(self, result.grad)
        _set_tensor_grad(other, -result.grad)

    result._backward_op = _backward
    result._prev = [self, other]
    return result


# [binary] element-wise multiplication
def _bfunc_mul(self: Tensor, other: Tensor) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)

    result = Tensor(
        self.data * other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def _backward() -> None:
        _set_tensor_grad(self, other.data * result.grad)
        _set_tensor_grad(other, self.data * result.grad)

    result._backward_op = _backward
    result._prev = [self, other]
    return result


# [binary] true-division
def _bfunc_truediv(self: Tensor, other: Tensor) -> Tensor:
    other = other if isinstance(other, Tensor) else Tensor(other)

    result = Tensor(
        self.data / other.data,
        requires_grad=self.requires_grad or other.requires_grad,
    )

    def _backward() -> None:
        _set_tensor_grad(self, (1 / other.data) * result.grad)
        _set_tensor_grad(other, -self.data / (other.data**2) * result.grad)

    result._backward_op = _backward
    result._prev = [self, other]
    return result


# [unary] power
def _ufunc_pow(self: Tensor, exp: float) -> Tensor:
    result = Tensor(self.data**exp, requires_grad=self.requires_grad)

    def _backward() -> None:
        _set_tensor_grad(self, (exp * self.data ** (exp - 1)) * result.grad)

    result._backward_op = _backward
    result._prev = [self]
    return result
