import numpy as np

from lucid.tensor import Tensor


__all__ = ("pow",)


def _set_tensor_grad(tensor: Tensor, grad: np.ndarray) -> None:
    if tensor.requires_grad:
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad += grad


def pow(self: Tensor, exp: float) -> Tensor:
    result = Tensor(self.data**exp, requires_grad=self.requires_grad)

    def _backward() -> None:
        _set_tensor_grad(self, (exp * self.data ** (exp - 1)) * result.grad)

    result._backward_op = _backward
    result._prev = [self]
    return result
