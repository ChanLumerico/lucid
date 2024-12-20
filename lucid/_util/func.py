from typing import Literal, Sequence
import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _NumPyArray, _ArrayLikeInt

from lucid._backend import (
    create_func_op,
    create_ufunc_op,
    create_mfunc_op,
    _FuncOpReturnType,
)


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


@create_ufunc_op()
def pad(self: Tensor, pad_width: _ArrayLikeInt) -> _FuncOpReturnType:
    result = Tensor(np.pad(self.data, pad_width))

    def _normalize_pad_width(pad_width: _ArrayLikeInt, ndim: int) -> _ArrayLikeInt:
        if isinstance(pad_width, int):
            return ((pad_width, pad_width),) * ndim

        if isinstance(pad_width, (tuple, list)):
            pad_width = list(pad_width)
            if all(isinstance(pw, int) for pw in pad_width):
                if len(pad_width) == 1:
                    return ((pad_width[0], pad_width[0]),) * ndim
                elif len(pad_width) == 2:
                    return (tuple(pad_width),) * ndim
                elif len(pad_width) == ndim:
                    return tuple((pw, pw) for pw in pad_width)

            elif all(
                isinstance(pw, (tuple, list)) and len(pw) == 2 for pw in pad_width
            ):
                if len(pad_width) == ndim:
                    return tuple(tuple(pw) for pw in pad_width)
                elif len(pad_width) == 1:
                    return (tuple(pad_width[0]),) * ndim

        raise ValueError(f"Invalid pad_width format: '{pad_width}'.")

    pad_width_norm = _normalize_pad_width(pad_width, self.ndim)

    def compute_grad() -> _NumPyArray:
        grad_input = np.zeros_like(self.data)
        slices = []
        for pw in pad_width_norm:
            before, after = pw
            start = before
            end = -after if after != 0 else None
            slices.append(slice(start, end))

        grad_input = result.grad[tuple(slices)]
        return grad_input

    return result, compute_grad


@create_ufunc_op()
def repeat(
    self: Tensor, repeats: int | Sequence[int], axis: int | None = None
) -> _FuncOpReturnType:
    result = Tensor(np.repeat(self.data, repeats, axis=axis))

    def compute_grad() -> _NumPyArray:
        grad_input = np.zeros_like(self.data)
        repeats_arr = np.asarray(repeats)

        if axis is None:
            input_flat = self.data.flatten()
            grad_input_flat = grad_input.flatten()
            grad_output_flat = result.grad.flatten()

            input_size = input_flat.size

            if repeats_arr.size == 1:
                repeats_arr = np.full(input_size, repeats_arr)
            elif repeats_arr.size != input_size:
                raise ValueError(
                    "repeats must be an integer or a "
                    + "sequence of the same length as input."
                )

            input_indices = np.arange(input_size)
            output_indices = np.repeat(input_indices, repeats_arr)

            np.add.at(grad_input_flat, output_indices, grad_output_flat)
            grad_input = grad_input_flat.reshape(self.shape)

        else:
            if repeats_arr.size == 1:
                repeats_arr = np.full(self.shape[axis], repeats_arr)
            elif repeats_arr.size != self.shape[axis]:
                raise ValueError(
                    "repeats must be an integer or a "
                    + "sequence of the same length as the axis dimension."
                )

            expand_dims = [1] * self.ndim
            expand_dims[axis] = -1

            input_indices_axis = np.arange(self.shape[axis]).reshape(expand_dims)
            output_indices_axis = np.repeat(input_indices_axis, repeats_arr, axis=axis)

            idx = np.indices(result.grad.shape)
            idx[axis] = output_indices_axis

            np.add.at(grad_input, tuple(idx), result.grad)

        return grad_input

    return result, compute_grad


@create_ufunc_op()
def tile(self: Tensor, reps: int | Sequence[int]) -> _FuncOpReturnType:
    result = Tensor(np.tile(self.data, reps))

    def compute_grad() -> _NumPyArray:
        if self.ndim == 0:
            input_shape = (1,)
            if isinstance(reps, int):
                reps_list = (reps,)
            else:
                reps_list = tuple(reps)
                if len(reps_list) == 0:
                    reps_list = (1,)
        else:
            input_shape = np.array(self.shape)
            if isinstance(reps, int):
                reps_list = (1,) * (self.ndim - 1) + (reps,)
            else:
                reps_list = tuple(reps)
                if len(reps_list) < self.ndim:
                    reps_list = (1,) * (self.ndim - len(reps_list)) + reps_list

        reps_array = np.array(reps_list)

        reshape_dims = []
        for dim_size, rep in zip(input_shape, reps_array):
            reshape_dims.extend([rep, dim_size])

        grad_output = result.grad
        if grad_output.size != np.prod(reshape_dims):
            raise ValueError(
                f"Cannot reshape array of size {grad_output.size} "
                + f"into shape {reshape_dims}"
            )

        grad_output_reshape = grad_output.reshape(reshape_dims)
        axes_to_sum = tuple(range(0, grad_output_reshape.ndim, 2))

        return grad_output_reshape.sum(axis=axes_to_sum)

    return result, compute_grad


@create_ufunc_op()
def flatten(self: Tensor) -> _FuncOpReturnType:
    original_shape = self.shape
    result = Tensor(self.data.reshape(-1))

    def compute_grad() -> _NumPyArray:
        return result.grad.reshape(*original_shape)

    return result, compute_grad


@create_func_op(n_in=2, n_ret=2)
def meshgrid(
    self: Tensor, other: Tensor, indexing: Literal["xy", "ij"]
) -> _FuncOpReturnType:
    if self.ndim != 1 or other.ndim != 1:
        raise ValueError("Inputs must be 1D tensors.")

    if indexing not in {"xy", "ij"}:
        raise ValueError("indexing must be either 'xy' or 'ij'")

    X = self.reshape(1, -1).repeat(other.shape[0], axis=0)
    Y = other.reshape(-1, 1).repeat(self.shape[0], axis=1)

    if indexing == "xy":
        X, Y = Y, X

    def compute_grad() -> tuple[_NumPyArray, _NumPyArray]:
        grad_x = np.sum(X.grad, axis=0)
        grad_y = np.sum(Y.grad, axis=1)
        return grad_x, grad_y

    return (X, compute_grad), (Y, compute_grad)
