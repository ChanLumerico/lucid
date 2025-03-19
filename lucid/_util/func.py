from typing import Literal, Sequence
from types import ModuleType
from functools import partial

import numpy as np
import math

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _NumPyArray, _ArrayLikeInt, _Scalar

from lucid._backend.core import (
    operation,
    func_op,
    unary_func_op,
    poly_func_op,
    _FuncOpReturnType,
    _GradFuncType,
)
from lucid._backend.metal import mx


class reshape(operation):
    def __init__(self, shape: _ShapeLike) -> None:
        super().__init__()
        self.shape = shape

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.reshape(self.shape))
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.reshape(a.data, self.shape))
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return self.result.grad.reshape(*a.shape)


class _reshape_immediate(operation):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.reshape(self.shape))
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.reshape(a.data, self.shape))
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return self.result.grad.reshape(a.shape)


class squeeze(operation):
    def __init__(self, axis: _ShapeLike | None = None) -> None:
        super().__init__()
        self.axis = axis

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.squeeze(axis=self.axis))
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.squeeze(a.data, axis=self.axis))
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return self.result.grad.reshape(a.shape)


class unsqueeze(operation):
    def __init__(self, axis: _ShapeLike) -> None:
        super().__init__()
        self.axis = axis

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.expand_dims(a.data, axis=self.axis))
        return self.result, partial(self.compute_grad, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.expand_dims(a.data, axis=self.axis))
        return self.result, partial(self.compute_grad, lib_=mx)

    def compute_grad(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.squeeze(self.result.grad, axis=self.axis)


class expand_dims(unsqueeze): ...


class ravel(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.ravel())
        return self.result, partial(self.compute_grad, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.reshape(a.data, (-1,)))
        return self.result, partial(self.compute_grad, a=a)

    def compute_grad(self, a: Tensor) -> _GradFuncType:
        return self.result.grad.reshape(a.shape)


class stack(operation):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data for a in arr]
        self.result = Tensor(np.stack(data_arr, axis=self.axis))

        return self.result, partial(self.compute_grad, arr=arr, lib_=np)

    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [t.data for t in arr]
        self.result = Tensor(mx.stack(data_arr, axis=self.axis))

        return self.result, partial(self.compute_grad, arr=arr, lib_=mx)

    def compute_grad(self, arr: tuple[Tensor], lib_: ModuleType) -> _GradFuncType:
        split_grads = lib_.split(self.result.grad, len(arr), axis=self.axis)
        return tuple(split_grads)


class hstack(stack):
    def __init__(self) -> None:
        super().__init__(axis=1)

    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data if a.ndim > 1 else a.data.reshape(-1, 1) for a in arr]
        self.result = Tensor(np.hstack(data_arr))

        return self.result, partial(self.compute_grad, arr=arr, lib_=np)

    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [t.data if t.ndim > 1 else t.data.reshape(-1, 1) for t in arr]
        self.result = Tensor(mx.concatenate(data_arr, axis=1))

        return self.result, partial(self.compute_grad, arr=arr, lib_=mx)


class vstack(stack):
    def __init__(self) -> None:
        super().__init__(axis=0)

    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data if a.ndim > 1 else a.data.reshape(1, -1) for a in arr]
        self.result = Tensor(np.vstack(data_arr))

        return self.result, partial(self.compute_grad, arr=arr, lib_=np)

    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [t.data if t.ndim > 1 else t.data.reshape(1, -1) for t in arr]
        self.result = Tensor(mx.concatenate(data_arr, axis=0))

        return self.result, partial(self.compute_grad, arr=arr, lib_=mx)


class concatenate(operation):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data for a in arr]
        self.result = Tensor(np.concatenate(data_arr, axis=self.axis))
        return self.result, partial(self.compute_grad, arr=arr, lib_=np)

    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data for a in arr]
        self.result = Tensor(mx.concatenate(data_arr, axis=self.axis))
        return self.result, partial(self.compute_grad, arr=arr, lib_=mx)

    def compute_grad(self, arr: Tensor, lib_: ModuleType) -> _GradFuncType:
        split_sizes = [a.shape[self.axis] for a in arr]
        split_indices = lib_.cumsum(lib_.array(split_sizes))[:-1]

        split_grad = lib_.split(self.result.grad, split_indices, axis=self.axis)
        return tuple(split_grad)


class pad(operation):
    def __init__(self, pad_width: _ArrayLikeInt, ndim: int) -> None:
        super().__init__()
        self.pad_width = pad_width
        self.pad_with_norm = self._normalize_pad_width(pad_width, ndim)

    def _normalize_pad_width(
        self, pad_width: _ArrayLikeInt, ndim: int
    ) -> _ArrayLikeInt:
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

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.pad(a.data, self.pad_width))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.pad(a.data, self.pad_width))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        grad_input = lib_.zeros_like(a.data)
        slices = []
        for pw in self.pad_with_norm:
            before, after = pw
            start = before
            end = -after if after != 0 else None
            slices.append(slice(start, end))

        grad_input = self.result.grad[tuple(slices)]
        return grad_input


class repeat(operation):
    def __init__(self, repeats: int | Sequence[int], axis: int | None) -> None:
        super().__init__()
        self.repeats = repeats
        self.axis = axis

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.repeat(a.data, self.repeats, axis=self.axis))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.repeat(a.data, self.repeats, axis=self.axis))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        grad_input = lib_.zeros_like(a.data)
        repeats_arr = lib_.array(self.repeats)

        if self.axis is None:
            input_flat = a.data.flatten()
            grad_input_flat = grad_input.flatten()
            grad_output_flat = self.result.grad.flatten()

            input_size = input_flat.size

            if repeats_arr.size == 1:
                repeats_arr = lib_.full(input_size, repeats_arr)
            elif repeats_arr.size != input_size:
                raise ValueError(
                    "repeats must be an integer or a "
                    + "sequence of the same length as input."
                )

            input_indices = lib_.arange(input_size)
            if lib_ is np:
                output_indices = np.repeat(input_indices, repeats_arr)
                np.add.at(grad_input_flat, output_indices, grad_output_flat)
            else:
                output_indices = mx.concatenate(
                    [
                        mx.full((r,), idx)
                        for idx, r in zip(input_indices.tolist(), repeats_arr)
                    ]
                )
                grad_input_flat = grad_input_flat.at[output_indices].add(
                    grad_output_flat
                )

            grad_input = grad_input_flat.reshape(a.shape)

        else:
            axis_ = self.axis % a.ndim
            if repeats_arr.size == 1:
                repeats_arr = lib_.full(a.shape[self.axis], repeats_arr)
            elif repeats_arr.size != a.shape[self.axis]:
                raise ValueError(
                    "repeats must be an integer or a "
                    + "sequence of the same length as the axis dimension."
                )

            expand_dims = [1] * a.ndim
            expand_dims[axis_] = -1

            input_indices_axis = lib_.arange(a.shape[axis_]).reshape(expand_dims)
            if lib_ is np:
                output_indices_axis = np.repeat(
                    input_indices_axis, repeats_arr, axis=axis_
                )
            else:
                moved_input_ = mx.moveaxis(input_indices_axis, axis_, 0)
                input_shape_ = input_indices_axis.shape
                slices = [
                    mx.full(
                        (*input_shape_[:axis_], r, *input_shape_[axis_ + 1 :]),
                        mx.expand_dims(slice_, axis_),
                    )
                    for slice_, r in zip(moved_input_, repeats_arr)
                ]
                output_indices_axis = mx.concatenate(slices, axis=axis_)

            idx = lib_.stack(
                lib_.meshgrid(
                    *[lib_.arange(s) for s in self.result.grad.shape], indexing="ij"
                )
            )
            idx[axis_] = output_indices_axis

            if lib_ is np:
                np.add.at(grad_input, tuple(idx), self.result.grad)
            else:
                grad_input = grad_input.at[tuple(idx)].add(self.result.grad)

        return grad_input


class tile(operation):
    def __init__(self, reps: int | Sequence[int]) -> None:
        super().__init__()
        self.reps = reps

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.tile(a.data, self.reps))
        return self.result, partial(self.compute_grad, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.tile(a.data, self.reps))
        return self.result, partial(self.compute_grad, a=a, lib_=mx)

    def compute_grad(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
        if a.ndim == 0:
            input_shape = (1,)
            if isinstance(self.reps, int):
                reps_list = (self.reps,)
            else:
                reps_list = tuple(self.reps)
                if len(reps_list) == 0:
                    reps_list = (1,)
        else:
            input_shape = lib_.array(a.shape)
            if isinstance(self.reps, int):
                reps_list = (1,) * (a.ndim - 1) + (self.reps,)
            else:
                reps_list = tuple(self.reps)
                if len(reps_list) < self.ndim:
                    reps_list = (1,) * (a.ndim - len(reps_list)) + reps_list

        reps_array = lib_.array(reps_list)

        reshape_dims = []
        for dim_size, rep in zip(input_shape, reps_array):
            reshape_dims.extend([rep, dim_size])

        grad_output = self.result.grad
        if grad_output.size != lib_.prod(lib_.array(reshape_dims)):
            raise ValueError(
                f"Cannot reshape array of size {grad_output.size} "
                + f"into shape {reshape_dims}"
            )

        grad_output_reshape = grad_output.reshape(reshape_dims)
        axes_to_sum = tuple(range(0, grad_output_reshape.ndim, 2))

        return grad_output_reshape.sum(axis=axes_to_sum)


class flatten(operation):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.original_shape = a.shape
        self.result = Tensor(a.data.reshape(*a.shape[: self.axis], -1))
        return self.result, self.compute_grad

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.original_shape = a.shape
        self.result = Tensor(a.data.reshape(*a.shape[: self.axis], -1))
        return self.result, self.compute_grad

    def compute_grad(self) -> _GradFuncType:
        return self.result.grad.reshape(self.original_shape)


# TODO: Continue from here


@func_op(n_in=2, n_ret=2)
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


def split(
    self: Tensor, size_or_sections: int | list[int] | tuple[int], axis: int = 0
) -> tuple[Tensor, ...]:
    returns = []
    if axis < 0:
        axis = self.ndim + axis

    self.axislen = self.shape[axis]
    if isinstance(size_or_sections, int):
        size_or_sections = (size_or_sections,) * int(
            math.ceil(self.axislen / size_or_sections)
        )

    cur_idx = 0
    for size in size_or_sections:
        slices = []
        for _ in range(axis):
            slices.append(slice(None, None, None))

        slices.append(slice(cur_idx, cur_idx + size, None))
        returns.append(self[*slices])
        cur_idx += size

    return tuple(returns)


@unary_func_op()
def tril(self: Tensor, diagonal: int = 0) -> _FuncOpReturnType:
    result = Tensor(np.tril(self.data, k=diagonal))

    def compute_grad() -> _NumPyArray:
        return np.tril(result.grad, k=diagonal)

    return result, compute_grad


@unary_func_op()
def triu(self: Tensor, diagonal: int = 0) -> _FuncOpReturnType:
    result = Tensor(np.triu(self.data, k=diagonal))

    def compute_grad() -> _NumPyArray:
        return np.triu(result.grad, k=diagonal)

    return result, compute_grad


@unary_func_op()
def broadcast_to(self: Tensor, shape: _ShapeLike) -> _FuncOpReturnType:
    original_shape = self.shape
    result = Tensor(np.broadcast_to(self.data, shape))

    def compute_grad() -> _NumPyArray:
        input_shape = original_shape
        ndim_diff = len(shape) - len(input_shape)
        if ndim_diff > 0:
            input_shape = (1,) * ndim_diff + input_shape

        for axis, (in_dim, out_dim) in enumerate(zip(input_shape, shape)):
            if in_dim == 1 and out_dim > 1:
                result.grad = result.grad.sum(axis=axis, keepdims=True)

        return result.grad.reshape(original_shape)

    return result, compute_grad


@func_op(n_in=1, n_ret=None)
def chunk(self: Tensor, chunks: int, axis: int = 0) -> _FuncOpReturnType:
    if chunks <= 0:
        raise ValueError("chunks must be greater than 0.")

    dim_size = self.shape[axis]
    chunk_size = (dim_size + chunks - 1) // chunks

    split_indices = list(range(chunk_size, dim_size, chunk_size))
    chunked_arrays = np.split(self.data, split_indices, axis=axis)

    results = []
    start_idx = 0
    for arr in chunked_arrays:
        chunk_t = Tensor(arr)

        def compute_grad(_tensor: Tensor = chunk_t, _idx=start_idx) -> _NumPyArray:
            slices = [slice(None)] * self.ndim
            slices[axis] = slice(_idx, _idx + _tensor.shape[axis])

            grad = np.zeros_like(self.data)
            grad[tuple(slices)] = _tensor.grad

            return grad

        results.append((chunk_t, compute_grad))
        start_idx += chunk_t.shape[axis]

    return tuple(results)


@unary_func_op()
def masked_fill(self: Tensor, mask: Tensor, value: _Scalar) -> _FuncOpReturnType:
    mask = mask.astype(bool)
    result = Tensor(np.where(mask.data, value, self.data))

    def compute_grad() -> _NumPyArray:
        grad = result.grad.copy()
        grad[mask.data] = 0
        return grad

    return result, compute_grad


@unary_func_op()
def roll(
    self: Tensor,
    shifts: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> _FuncOpReturnType:
    result = Tensor(np.roll(self.data, shift=shifts, axis=axis))

    def compute_grad() -> _NumPyArray:
        if isinstance(shifts, int):
            neg_shifts = -shifts
        elif isinstance(shifts, tuple):
            neg_shifts = tuple(-s for s in shifts)

        return np.roll(result.grad, shift=neg_shifts, axis=axis)

    return result, compute_grad
