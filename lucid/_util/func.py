import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _NumPyArray

from lucid._backend import create_ufunc_op, create_mfunc_op, _FuncOpReturnType


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


@create_mfunc_op()
def stack(*tensors: Tensor, axis: int = 0) -> _FuncOpReturnType:
    data_arr = [tensor.data for tensor in tensors]
    result = Tensor(np.stack(data_arr, axis=axis))

    def compute_grad() -> tuple[_NumPyArray, ...]:
        split_grads = np.split(result.grad, len(tensors), axis=axis)
        return tuple(split_grads)

    return result, compute_grad


@create_mfunc_op()
def concatenate(*tensors: Tensor, axis: int = 0) -> _FuncOpReturnType:
    data_arr = [tensor.data for tensor in tensors]
    result = Tensor(np.concatenate(data_arr, axis=axis))

    def compute_grad() -> tuple[_NumPyArray, ...]:
        split_sizes = [tensor.shape[axis] for tensor in tensors]
        split_indices = np.cumsum(split_sizes)[:-1]
        split_grads = np.split(result.grad, split_indices, axis=axis)

        return tuple(split_grads)

    return result, compute_grad


@create_mfunc_op()
def hstack(*tensors: Tensor) -> _FuncOpReturnType:
    data_arr = [tensor.data for tensor in tensors]
    result = Tensor(np.hstack(data_arr))

    def compute_grad() -> tuple[_NumPyArray, ...]:
        split_sizes = [
            tensor.shape[1] if result.ndim > 1 else tensor.shape[0]
            for tensor in tensors
        ]
        split_indices = np.cumsum(split_sizes)[:-1]
        split_grads = (
            np.hsplit(result.grad, split_indices)
            if result.ndim > 1
            else np.split(result.grad, len(tensors))
        )

        return tuple(split_grads)

    return result, compute_grad


@create_mfunc_op()
def vstack(*tensors: Tensor) -> _FuncOpReturnType:
    data_arr = [tensor.data for tensor in tensors]
    result = Tensor(np.vstack(data_arr))

    def compute_grad() -> tuple[_NumPyArray, ...]:
        split_sizes = [tensor.shape[0] for tensor in tensors]
        split_indices = np.cumsum(split_sizes)[:-1]
        split_grads = np.split(result.grad, split_indices, axis=0)

        return tuple(split_grads)

    return result, compute_grad
