from typing import Literal, Sequence, override
from types import ModuleType
from functools import partial

import numpy as np
import math

import lucid
from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLikeInt, _Scalar, _ArrayLike, _NumPyArray

from lucid._backend.core import (
    fallback,
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
        return self.result, partial(self.__grad__, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.reshape(a.data, self.shape))
        return self.result, partial(self.__grad__, a=a)

    def __grad__(self, a: Tensor) -> _GradFuncType:
        return self.result.grad.reshape(*a.shape)


class _reshape_immediate(operation):
    def __init__(self, shape: _ShapeLike) -> None:
        super().__init__()
        self.shape = shape

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.reshape(self.shape))
        return self.result, partial(self.__grad__, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.reshape(a.data, self.shape))
        return self.result, partial(self.__grad__, a=a)

    def __grad__(self, a: Tensor) -> _GradFuncType:
        return self.result.grad.reshape(a.shape)


class squeeze(operation):
    def __init__(self, axis: _ShapeLike | None = None) -> None:
        super().__init__()
        self.axis = axis

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.squeeze(axis=self.axis))
        return self.result, partial(self.__grad__, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.squeeze(a.data, axis=self.axis))
        return self.result, partial(self.__grad__, a=a)

    def __grad__(self, a: Tensor) -> _GradFuncType:
        return self.result.grad.reshape(a.shape)


class unsqueeze(operation):
    def __init__(self, axis: _ShapeLike) -> None:
        super().__init__()
        self.axis = axis

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.expand_dims(a.data, axis=self.axis))
        return self.result, partial(self.__grad__, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.expand_dims(a.data, axis=self.axis))
        return self.result, partial(self.__grad__, lib_=mx)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.squeeze(self.result.grad, axis=self.axis)


class expand_dims(unsqueeze): ...


class ravel(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(a.data.ravel())
        return self.result, partial(self.__grad__, a=a)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.reshape(a.data, (-1,)))
        return self.result, partial(self.__grad__, a=a)

    def __grad__(self, a: Tensor) -> _GradFuncType:
        return self.result.grad.reshape(a.shape)


class stack(operation):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data for a in arr]
        self.result = Tensor(np.stack(data_arr, axis=self.axis))

        return self.result, partial(self.__grad__, arr=arr, lib_=np)

    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [t.data for t in arr]
        self.result = Tensor(mx.stack(data_arr, axis=self.axis))

        return self.result, partial(self.__grad__, arr=arr, lib_=mx)

    def __grad__(self, arr: tuple[Tensor], lib_: ModuleType) -> _GradFuncType:
        split_grads = lib_.split(self.result.grad, len(arr), axis=self.axis)
        return tuple(split_grads)


class hstack(stack):
    def __init__(self) -> None:
        super().__init__(axis=1)

    @override
    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data if a.ndim > 1 else a.data.reshape(-1, 1) for a in arr]
        self.result = Tensor(np.hstack(data_arr))

        return self.result, partial(self.__grad__, arr=arr, lib_=np)

    @override
    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [t.data if t.ndim > 1 else t.data.reshape(-1, 1) for t in arr]
        self.result = Tensor(mx.concatenate(data_arr, axis=1))

        return self.result, partial(self.__grad__, arr=arr, lib_=mx)


class vstack(stack):
    def __init__(self) -> None:
        super().__init__(axis=0)

    @override
    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data if a.ndim > 1 else a.data.reshape(1, -1) for a in arr]
        self.result = Tensor(np.vstack(data_arr))

        return self.result, partial(self.__grad__, arr=arr, lib_=np)

    @override
    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [t.data if t.ndim > 1 else t.data.reshape(1, -1) for t in arr]
        self.result = Tensor(mx.concatenate(data_arr, axis=0))

        return self.result, partial(self.__grad__, arr=arr, lib_=mx)


class concatenate(operation):
    def __init__(self, axis: int) -> None:
        super().__init__()
        self.axis = axis

    @poly_func_op()
    def cpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data for a in arr]
        self.result = Tensor(np.concatenate(data_arr, axis=self.axis))
        return self.result, partial(self.__grad__, arr=arr)

    @poly_func_op(device="gpu")
    def gpu(self, *arr: Tensor) -> _FuncOpReturnType:
        data_arr = [a.data for a in arr]
        self.result = Tensor(mx.concatenate(data_arr, axis=self.axis))
        return self.result, partial(self.__grad__, arr=arr)

    def __grad__(self, arr: tuple[Tensor, ...]) -> tuple:
        split_sizes = [a.shape[self.axis] for a in arr]
        grad = self.result.grad
        outputs = []
        start = 0

        for size in split_sizes:
            end = start + size
            slicer = [slice(None)] * grad.ndim
            slicer[self.axis] = slice(start, end)

            outputs.append(grad[tuple(slicer)])
            start = end

        return tuple(outputs)


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
        return self.result, partial(self.__grad__, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.pad(a.data, self.pad_width))
        return self.result, partial(self.__grad__, a=a, lib_=mx)

    def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
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
        return self.result, partial(self.__grad__, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.repeat(a.data, self.repeats, axis=self.axis))
        return self.result, partial(self.__grad__, a=a, lib_=mx)

    def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
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
        return self.result, partial(self.__grad__, a=a, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.tile(a.data, self.reps))
        return self.result, partial(self.__grad__, a=a, lib_=mx)

    def __grad__(self, a: Tensor, lib_: ModuleType) -> _GradFuncType:
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
        return self.result, self.__grad__

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.original_shape = a.shape
        self.result = Tensor(a.data.reshape(*a.shape[: self.axis], -1))
        return self.result, self.__grad__

    def __grad__(self) -> _GradFuncType:
        return self.result.grad.reshape(self.original_shape)


class meshgrid(operation):
    def __init__(self, indexing: Literal["xy", "ij"]) -> None:
        super().__init__()
        if indexing not in {"xy", "ij"}:
            raise ValueError("indexing must be either 'xy' or 'ij'")
        self.indexing = indexing

    def _unified(self, a: Tensor, b: Tensor, lib_: ModuleType) -> tuple[Tensor, Tensor]:
        if a.ndim != 1 or a.ndim != 1:
            raise ValueError(f"Input tensors must be 1D tensors.")

        X = lib_.repeat(a.data.reshape(1, -1), b.data.shape[0], axis=0)
        Y = lib_.repeat(b.data.reshape(-1, 1), a.data.shape[0], axis=1)

        if self.indexing == "xy":
            X, Y = Y, X

        return Tensor(X), Tensor(Y)

    @func_op(n_in=2, n_ret=2)
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.X, self.Y = self._unified(a, b, lib_=np)
        self.result = (self.X, self.Y)

        compute_grad = partial(self.__grad__, lib_=np)
        return (self.X, compute_grad), (self.Y, compute_grad)

    @func_op(n_in=2, n_ret=2, device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        self.X, self.Y = self._unified(a, b, lib_=mx)
        self.result = (self.X, self.Y)

        compute_grad = partial(self.__grad__, lib_=mx)
        return (self.X, compute_grad), (self.Y, compute_grad)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        grad_x = lib_.sum(self.X.grad, axis=0)
        grad_y = lib_.sum(self.Y.grad, axis=1)

        return grad_x, grad_y


class split(operation):
    def __init__(
        self, size_or_sections: int | list[int] | tuple[int], axis: int
    ) -> None:
        super().__init__()
        self.size_or_sections = size_or_sections
        self.axis = axis

    def cpu(self, *args, **kwargs) -> _FuncOpReturnType:
        return super().cpu(*args, **kwargs)

    def gpu(self, *args, **kwargs) -> _FuncOpReturnType:
        return super().gpu(*args, **kwargs)

    @override
    def __call__(self, a: Tensor) -> tuple[Tensor, ...]:
        returns = []
        if self.axis < 0:
            self.axis = a.ndim + self.axis

        self.axislen = a.shape[self.axis]
        if isinstance(self.size_or_sections, int):
            self.size_or_sections = (self.size_or_sections,) * int(
                math.ceil(self.axislen / self.size_or_sections)
            )

        cur_idx = 0
        for size in self.size_or_sections:
            slices = []
            for _ in range(self.axis):
                slices.append(slice(None, None, None))

            slices.append(slice(cur_idx, cur_idx + size, None))
            returns.append(a[*slices])
            cur_idx += size

        self.result = tuple(returns)
        return self.result


class tril(operation):
    def __init__(self, diagonal: int) -> None:
        super().__init__()
        self.diagonal = diagonal

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.tril(a.data, k=self.diagonal))
        return self.result, partial(self.__grad__, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.tril(a.data, k=self.diagonal))
        return self.result, partial(self.__grad__, lib_=mx)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.tril(self.result.grad, k=self.diagonal)


class triu(operation):
    def __init__(self, diagonal: int) -> None:
        super().__init__()
        self.diagonal = diagonal

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.triu(a.data, k=self.diagonal))
        return self.result, partial(self.__grad__, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.triu(a.data, k=self.diagonal))
        return self.result, partial(self.__grad__, lib_=mx)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.triu(self.result.grad, k=self.diagonal)


class broadcast_to(operation):
    def __init__(self, shape: _ShapeLike) -> None:
        super().__init__()
        self.shape = shape

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.original_shape = a.shape
        self.result = Tensor(np.broadcast_to(a.data, self.shape))

        return self.result, self.__grad__

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.original_shape = a.shape
        self.result = Tensor(mx.broadcast_to(a.data, self.shape))

        return self.result, self.__grad__

    def __grad__(self) -> _GradFuncType:
        input_shape = self.original_shape
        ndim_diff = len(self.shape) - len(input_shape)
        if ndim_diff > 0:
            input_shape = (1,) * ndim_diff + input_shape

        for axis, (in_dim, out_dim) in enumerate(zip(input_shape, self.shape)):
            if in_dim == 1 and out_dim > 1:
                self.result.grad = self.result.grad.sum(axis=axis, keepdims=True)

        return self.result.grad.reshape(self.original_shape)


class chunk(operation):
    def __init__(self, chunks: int, axis: int) -> None:
        super().__init__()
        self.chunks = chunks
        self.axis = axis

    def _unified(self, a: Tensor, lib_: ModuleType) -> _FuncOpReturnType:
        if self.chunks <= 0:
            raise ValueError("chunks must be greater than 0.")

        dim_size = a.shape[self.axis]
        chunk_size = (dim_size + self.chunks - 1) // self.chunks

        split_indices = list(range(chunk_size, dim_size, chunk_size))
        chunked_arrays = lib_.split(a.data, split_indices, axis=self.axis)

        results = []
        start_idx = 0
        for arr in chunked_arrays:
            chunk_t = Tensor(arr)

            def compute_grad(_a: Tensor = chunk_t, _idx=start_idx) -> _GradFuncType:
                slices = [slice(None)] * a.ndim
                slices[self.axis] = slice(_idx, _idx + _a.shape[self.axis])

                grad = lib_.zeros_like(a.data)
                grad[tuple(slices)] = _a.grad

                return grad

            results.append((chunk_t, compute_grad))
            start_idx += chunk_t.shape[self.axis]

        return tuple(results)

    @func_op(n_in=1, n_ret=None)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._unified(a, lib_=np)

    @func_op(n_in=1, n_ret=None, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._unified(a, lib_=mx)


class masked_fill(operation):
    def __init__(self, mask: Tensor, value: _Scalar) -> None:
        super().__init__()
        self.mask = mask
        self.value = value

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.where(self.mask.data.astype(bool), self.value, a.data))
        return self.result, self.__grad_cpu__

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.where(self.mask.data.astype(bool), self.value, a.data))
        return self.result, self.__grad_gpu__

    def __grad_cpu__(self) -> _GradFuncType:
        grad = self.result.grad.copy()
        grad[self.mask.data] = 0
        return grad

    def __grad_gpu__(self) -> _GradFuncType:
        grad = mx.array(self.result.grad)
        grad = mx.where(self.mask.data.astype(bool), 0, grad)
        return grad


class roll(operation):
    def __init__(
        self, shifts: int | tuple[int, ...], axis: int | tuple[int, ...] | None
    ) -> None:
        super().__init__()
        self.shifts = shifts
        self.axis = axis

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(np.roll(a.data, shift=self.shifts, axis=self.axis))
        return self.result, partial(self.__grad__, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        self.result = Tensor(mx.roll(a.data, shift=self.shifts, axis=self.axis))
        return self.result, partial(self.__grad__, lib_=mx)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        if isinstance(self.shifts, int):
            neg_shift = -self.shifts
        elif isinstance(self.shifts, tuple):
            neg_shift = tuple(-s for s in self.shifts)

        return lib_.roll(self.result.grad, shift=neg_shift, axis=self.axis)


class unbind(operation):
    def __init__(self, axis: int = 0) -> None:
        super().__init__()
        self.axis = axis

    def _take_along_axis(self, x: _ArrayLike, index: int) -> _ArrayLike:
        idx = [slice(None)] * x.ndim
        idx[self.axis] = index
        return x[tuple(idx)]

    def _unified(self, a: Tensor, lib_: ModuleType) -> _FuncOpReturnType:
        dim_size = a.shape[self.axis]
        results = []

        for idx in range(dim_size):
            arr = self._take_along_axis(a.data, idx)
            ten = Tensor(arr)

            def compute_grad(_a: Tensor = ten, _idx: int = idx) -> _GradFuncType:
                idx_slices = [slice(None)] * a.ndim
                idx_slices[self.axis] = _idx

                grad = lib_.zeros_like(a.data)
                grad[tuple(idx_slices)] = _a.grad
                return grad

            results.append((ten, compute_grad))

        return tuple(results)

    @func_op(n_in=1, n_ret=None)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._unified(a, lib_=np)

    @func_op(n_in=1, n_ret=None, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        return self._unified(a, lib_=mx)


_SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]


class sort(operation):
    def __init__(
        self,
        axis: int = -1,
        descending: bool = False,
        kind: _SortKind | None = None,
        stable: bool = False,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.descending = descending
        self.stable = stable

        kind = kind or "quicksort"
        if self.stable:
            kind = "stable"
        self.kind = kind

    @func_op(n_in=1, n_ret=2)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        data = a.data
        indices = np.argsort(data, axis=self.axis, kind=self.kind)
        sorted_data = np.take_along_axis(data, indices, axis=self.axis)

        if self.descending:
            indices = np.flip(indices, axis=self.axis)
            sorted_data = np.take_along_axis(data, indices, axis=self.axis)

        values = Tensor(sorted_data)
        indices_t = Tensor(indices)

        self.result = (values, indices_t)
        return (
            (values, partial(self.__grad__, lib_=np)),
            (indices_t, lambda: np.zeros_like(indices)),
        )

    @func_op(n_in=1, n_ret=2, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        data = a.data
        indices = mx.argsort(data, axis=self.axis)

        if self.descending:
            shape = [1] * data.ndim
            shape[self.axis] = -1

            rev_idx = mx.arange(indices.shape[self.axis] - 1, -1, -1)
            rev_idx = rev_idx.reshape(*shape)
            indices = mx.take_along_axis(indices, rev_idx, axis=self.axis)

        sorted_data = mx.take_along_axis(data, indices, axis=self.axis)

        values = Tensor(sorted_data)
        indices_t = Tensor(indices)

        self.result = (values, indices_t)
        return (
            (values, partial(self.__grad__, lib_=mx)),
            (indices_t, lambda: mx.zeros_like(indices)),
        )

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        grad = self.result[0].grad
        reverse_indices = lib_.argsort(self.result[1].data, axis=self.axis)

        grad_out = lib_.take_along_axis(grad, reverse_indices, axis=self.axis)
        return grad_out

    def __flops__(self, a: Tensor) -> int:
        axis = self.axis if self.axis >= 0 else a.ndim + self.axis
        n = a.shape[axis]

        num_slices = math.prod([s for i, s in enumerate(a.shape) if i != axis])
        return int(num_slices * n * math.log2(max(n, 2)))


class argsort(operation):
    def __init__(
        self,
        axis: int = -1,
        descending: bool = False,
        kind: _SortKind | None = None,
        stable: bool = False,
    ) -> None:
        super().__init__()
        self.axis = axis
        self.descending = descending
        self.stable = stable

        kind = kind or "quicksort"
        if self.stable:
            kind = "stable"
        self.kind = kind

    @unary_func_op(has_gradient=False)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        axis = self.axis if self.axis >= 0 else a.ndim + self.axis
        data = -a.data if self.descending else a.data
        indices = np.argsort(data, axis=axis, kind=self.kind)

        self.result = Tensor(indices.astype(np.int32))
        return self.result, partial(self.__grad__, lib_=np)

    @unary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        axis = self.axis if self.axis >= 0 else a.ndim + self.axis
        data = mx.negative(a.data) if self.descending else a.data
        indices = mx.argsort(data, axis=axis)

        self.result = Tensor(indices.astype(mx.int32))
        return self.result, partial(self.__grad__, lib_=mx)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0)

    def __flops__(self, a: Tensor) -> int:
        axis = self.axis if self.axis >= 0 else a.ndim + self.axis
        n = a.shape[axis]
        num_slices = math.prod([s for i, s in enumerate(a.shape) if i != axis])
        return int(num_slices * n * math.log2(max(n, 2)))


class nonzero(operation):
    def __init__(self) -> None:
        super().__init__()

    @unary_func_op(has_gradient=False)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        coords = np.transpose(np.nonzero(a.data))
        self.result = Tensor(coords.astype(np.int32))
        return self.result, partial(self.__grad__, lib_=np)

    @unary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        shape = a.shape
        ndim = a.ndim
        size = a.size

        mask = mx.not_equal(a.data, mx.zeros_like(a.data)).reshape(-1)
        flat_idx = mx.arange(size)

        marked = mx.where(mask, flat_idx, size)
        sorted_idx = mx.sort(marked)
        count = int(mx.sum(mx.less(sorted_idx, size)).item())
        kept = sorted_idx[:count]

        strides = []
        acc = 1
        for s in reversed(shape):
            strides.append(acc)
            acc *= s

        strides = strides[::-1]
        strides_tensor = mx.array(strides).reshape(1, ndim)

        kept = kept.reshape(-1, 1)
        coords = (kept // strides_tensor) % mx.array(shape).reshape(1, ndim)
        coords = coords.astype(mx.int32)

        self.result = Tensor(coords)
        return self.result, partial(self.__grad__, lib_=mx)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0)

    def __flops__(self, a: Tensor) -> int:
        flat = a.data.flatten()
        return sum(bool(x) for x in flat)


@fallback
class unique(operation):
    def __init__(self, sorted: bool = True, axis: int | None = None) -> None:
        super().__init__()
        self.sorted = sorted
        self.axis = axis
        self.inverse_ = None

    def _unified(self, a: Tensor) -> _NumPyArray:
        data = a.data

        if self.sorted:
            unique_data, inverse = np.unique(data, axis=self.axis, return_inverse=True)
        else:
            unique_data, idx = np.unique(data, return_index=True, axis=self.axis)
            sorter = np.sort(idx)

            if self.axis is None:
                flat_data = np.ravel(data)
                unique_data = flat_data[sorter]
                _, inverse = np.unique(flat_data, return_inverse=True)
            else:
                unique_data = np.take(data, sorter, axis=self.axis)
                inverse = None

        self.inverse_ = None if inverse is None else Tensor(inverse)
        return unique_data

    @unary_func_op(has_gradient=False)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        out = self._unified(a)
        self.result = Tensor(out)
        return self.result, partial(self.__grad__, lib_=np)

    @unary_func_op(has_gradient=False, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        out = self._unified(a)
        self.result = Tensor(mx.array(out))
        if self.inverse_ is not None:
            self.inverse_ = Tensor(mx.array(self.inverse_.data))
        return self.result, partial(self.__grad__, lib_=mx)

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0)

    def __flops__(self, a: Tensor) -> int:
        return int(a.size * math.log2(max(a.size, 2)))


class topk(operation):
    def __init__(
        self, k: int, axis: int = -1, largest: bool = True, sorted: bool = True
    ) -> None:
        super().__init__()
        self.k = k
        self.axis = axis
        self.largest = largest
        self.sorted = sorted

    @func_op(n_in=1, n_ret=2)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        axis = self.axis if self.axis >= 0 else a.ndim + self.axis
        data = a.data.copy()
        negate_for_sort = not self.largest

        if negate_for_sort:
            data = -data

        indices = np.argpartition(data, -self.k, axis=axis)[..., -self.k :]
        values = np.take_along_axis(data, indices, axis=axis)

        if negate_for_sort:
            values = -values
            data = -data

        if self.sorted:
            sort_order = (
                np.argsort(-values, axis=axis)
                if self.largest
                else np.argsort(values, axis=axis)
            )
            values = np.take_along_axis(values, sort_order, axis=axis)
            indices = np.take_along_axis(indices, sort_order, axis=axis)

        values_t = Tensor(values)
        indices_t = Tensor(indices.astype(np.int32))

        self.result = (values_t, indices_t)
        return (
            (values_t, partial(self.__grad_cpu__, a=a)),
            (indices_t, lambda: np.zeros_like(indices)),
        )

    @func_op(n_in=1, n_ret=2, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        axis = self.axis if self.axis >= 0 else a.ndim + self.axis
        data = a.data
        negate_for_sort = not self.largest

        if negate_for_sort:
            data = mx.negative(data)

        indices = mx.argsort(data, axis=axis)
        indices = mx.take(indices, mx.arange(-self.k, 0), axis=axis)
        values = mx.take_along_axis(data, indices, axis=axis)

        if negate_for_sort:
            values = mx.negative(values)
            data = mx.negative(data)

        if self.sorted:
            sort_order = (
                mx.argsort(mx.negative(values), axis=axis)
                if self.largest
                else mx.argsort(values, axis=axis)
            )
            values = mx.take_along_axis(values, sort_order, axis=axis)
            indices = mx.take_along_axis(indices, sort_order, axis=axis)

        values_t = Tensor(values)
        indices_t = Tensor(indices.astype(mx.int32))

        self.result = (values_t, indices_t)
        return (
            (values_t, partial(self.__grad_gpu__, a=a)),
            (indices_t, lambda: mx.zeros_like(indices)),
        )

    def __grad_cpu__(self, a: Tensor) -> _GradFuncType:
        grad = self.result[0].grad
        axis = self.axis if self.axis >= 0 else grad.ndim + self.axis

        output_grad = np.zeros_like(a.data)
        np.put_along_axis(output_grad, self.result[1].data, grad, axis=axis)
        return output_grad

    def __grad_gpu__(self, a: Tensor) -> _GradFuncType:
        grad = self.result[0].grad
        axis = self.axis if self.axis >= 0 else grad.ndim + self.axis

        input_shape = a.shape
        indices = self.result[1].data

        B, k = indices.shape
        D = input_shape[axis]

        grad_exp = grad.reshape(B, k, 1)
        indices_exp = indices.reshape(B, k, 1)

        class_range = mx.arange(D).reshape(1, 1, D)
        one_hot = (indices_exp == class_range).astype(mx.float32)

        scatter = grad_exp * one_hot
        output = mx.sum(scatter, axis=1)
        if axis != 1:
            output = mx.moveaxis(output, 1, axis)

        return output

    def __flops__(self, a: Tensor) -> int:
        shape = a.shape
        axis = self.axis if self.axis >= 0 else len(shape) + self.axis

        n = shape[axis]
        num_slices = math.prod([s for i, s in enumerate(shape) if i != axis])

        if self.sorted:
            return int(num_slices * (n + self.k * math.log2(max(self.k, 1))))
        else:
            return int(num_slices * n)


class histogramdd(operation):
    def __init__(
        self,
        bins: list[int],
        range: list[tuple[float, float]],
        density: bool = False,
    ) -> None:
        super().__init__()
        if not all(isinstance(b, int) for b in bins):
            raise TypeError("All elements of bins must be integers.")

        self.bins = bins
        self.range = range
        self.density = density

    @func_op(n_in=1, n_ret=2, has_gradient=False)
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        hist, edges = np.histogramdd(a.data, self.bins, self.range, self.density)

        hist_t = Tensor(hist)
        edges_t = Tensor(np.stack([e for e in edges]))
        self.result = (hist_t, edges_t)
        return (
            (hist_t, partial(self.__grad__, lib_=np)),
            (edges_t, partial(self.__grad__, lib_=np)),
        )

    @func_op(n_in=1, n_ret=2, has_gradient=False, device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        data = a.data
        N, D = data.shape[:2]
        bins = self.bins

        edges = [mx.linspace(*r, bins[i] + 1) for i, r in enumerate(self.range)]
        bin_widths = [(r[1] - r[0]) / bins[i] for i, r in enumerate(self.range)]

        bin_indices = []
        for d in range(D):
            col = data[:, d]
            idx = mx.floor((col - self.range[d][0]) / bin_widths[d]).astype(mx.int32)
            idx = mx.clip(idx, 0, bins[d] - 1)
            bin_indices.append(idx)

        bin_indices = mx.stack(bin_indices, axis=1)
        hist_shape = tuple(bins)
        hist = mx.zeros(hist_shape, dtype=mx.int32)

        for i in range(N):
            idx = tuple(int(bin_indices[i, d].item()) for d in range(D))
            hist[idx] += 1

        hist = hist.astype(mx.float32)
        if self.density:
            total = mx.sum(hist)
            bin_volume = math.prod(bin_widths)
            hist /= total * bin_volume

        hist_t = Tensor(hist)
        edges_t = Tensor(mx.stack([e for e in edges]))
        self.result = (hist_t, edges_t)
        return (
            (hist_t, partial(self.__grad__, lib_=np)),
            (edges_t, partial(self.__grad__, lib_=np)),
        )

    def __grad__(self, lib_: ModuleType) -> _GradFuncType:
        return lib_.array(0.0)

    def __flops__(self, a: Tensor) -> int:
        return int(math.prod(a.shape) + math.prod(self.bins))
