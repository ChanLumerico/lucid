import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _NumPyArray

from lucid._backend import create_ufunc_op, _FuncOpReturnType


@create_ufunc_op()
def _reshape(self: Tensor, shape: _ShapeLike) -> _FuncOpReturnType:
    original_shape = self.shape
    result = Tensor(self.data.reshape(shape))

    def compute_grad() -> _NumPyArray:
        return result.grad.reshape(*original_shape)

    return result, compute_grad


@create_ufunc_op()
def _reshape_inplace(self: Tensor, *shape: int) -> _FuncOpReturnType:
    original_shape = self.shape
    result = Tensor(self.data.reshape(*shape))

    def compute_grad() -> _NumPyArray:
        return result.grad.reshape(*original_shape)

    return result, compute_grad


@create_ufunc_op()
def squeeze(self: Tensor, axis: _ShapeLike | None = None) -> _FuncOpReturnType:
    original_shape = self.shape
    result = Tensor(self.data.squeeze(axis=axis))

    def compute_grad() -> _NumPyArray:
        return result.grad.reshape(original_shape)

    return result, compute_grad


@create_ufunc_op()
def unsqueeze(self: Tensor, axis: _ShapeLike) -> _FuncOpReturnType:
    result = Tensor(np.expand_dims(self.data, axis=axis))

    def compute_grad() -> _NumPyArray:
        return result.grad.squeeze(axis=axis)

    return result, compute_grad


@create_ufunc_op()
def ravel(self: Tensor) -> _FuncOpReturnType:
    original_shape = self.shape
    result = Tensor(self.data.ravel())

    def compute_grad() -> _NumPyArray:
        return result.grad.reshape(original_shape)

    return result, compute_grad
