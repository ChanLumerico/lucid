import numpy as np

from lucid.tensor import Tensor


__all__ = ("add", "sub", "mul", "truediv")


def _set_tensor_grad(tensor: Tensor, grad: np.ndarray) -> None:
    if tensor.requires_grad:
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad += grad


def add(self: Tensor, other: Tensor) -> Tensor:
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


def sub(self: Tensor, other: Tensor) -> Tensor:
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


def mul(self: Tensor, other: Tensor) -> Tensor:
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


def truediv(self: Tensor, other: Tensor) -> Tensor:
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
