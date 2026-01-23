from functools import partial
from types import ModuleType
from typing import Literal, TypeAlias
import itertools

import numpy as np

from lucid._tensor import Tensor
from lucid._backend.core import (
    Operation,
    unary_func_op,
    _FuncOpReturnType,
    _GradType,
)
from lucid._backend.metal import mx
from lucid.types import _NumPyArray, _MLXArray


_Array: TypeAlias = _NumPyArray | _MLXArray
_Shape: TypeAlias = tuple[int, ...]
_Mode: TypeAlias = Literal["avg", "max"]


def _to_tuple(value: int | tuple[int, ...] | list[int], dim: int, name: str) -> _Shape:
    if isinstance(value, int):
        return (value,) * dim

    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return (int(value[0]),) * dim
        if len(value) != dim:
            raise ValueError(f"{name} must have length {dim}, got {len(value)}.")
        return tuple(int(v) for v in value)

    raise TypeError(f"{name} must be int or sequence, got {type(value).__name__}.")


def _prod(shape: _Shape) -> int:
    total = 1
    for v in shape:
        total *= int(v)
    return total


def _pool_out_dims(
    input_spatial: _Shape, kernel_size: _Shape, stride: _Shape, padding: _Shape
) -> _Shape:
    out_dims = []
    for i in range(len(kernel_size)):
        o = (input_spatial[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1
        if o <= 0:
            raise ValueError(f"Non-positive output dim for axis {i}: {o}")
        out_dims.append(o)
    return tuple(out_dims)


def _pad_input(lib_: ModuleType, data: _Array, padding: _Shape) -> _Array:
    if not any(padding):
        return data
    pad_width = ((0, 0), (0, 0)) + tuple((p, p) for p in padding)
    return lib_.pad(data, pad_width)


def _crop_padding(data: _Array, padding: _Shape) -> _Array:
    if not any(padding):
        return data

    slices = [slice(None), slice(None)]
    for p in padding:
        end = -p if p != 0 else None
        slices.append(slice(p, end))
    return data[tuple(slices)]


def _zeros(lib_: ModuleType, shape: _Shape, dtype: object) -> _Array:
    if lib_ is np:
        return np.zeros(shape, dtype=dtype)
    return mx.zeros(shape, dtype=dtype)


def _full_like_int(lib_: ModuleType, ref: _Array, value: int) -> _Array:
    shape = ref.shape
    if lib_ is np:
        return np.full(shape, value, dtype=np.int32)
    return mx.full(shape, value, dtype=mx.int32)


def _where(lib_: ModuleType, cond: _Array, x: _Array, y: _Array) -> _Array:
    if lib_ is np:
        return np.where(cond, x, y)
    return mx.where(cond, x, y)


def _pool_forward_sum(
    x_pad: _Array, out_dims: _Shape, kernel_size: _Shape, stride: _Shape
) -> _Array:
    out = None
    for k_idx in itertools.product(*[range(k) for k in kernel_size]):
        slices = [slice(None), slice(None)]
        for d in range(len(kernel_size)):
            start = k_idx[d]
            end = start + stride[d] * out_dims[d]
            slices.append(slice(start, end, stride[d]))

        window = x_pad[tuple(slices)]
        out = window if out is None else out + window

    return out


def _pool_forward_max(
    lib_: ModuleType,
    x_pad: _Array,
    out_dims: _Shape,
    kernel_size: _Shape,
    stride: _Shape,
) -> tuple[_Array, _Array]:
    max_vals = None
    max_idx = None

    for flat_idx, k_idx in enumerate(
        itertools.product(*[range(k) for k in kernel_size])
    ):
        slices = [slice(None), slice(None)]
        for d in range(len(kernel_size)):
            start = k_idx[d]
            end = start + stride[d] * out_dims[d]
            slices.append(slice(start, end, stride[d]))

        window = x_pad[tuple(slices)]
        if max_vals is None:
            max_vals = window
            max_idx = _full_like_int(lib_, window, flat_idx)
            continue

        mask = window > max_vals
        max_vals = _where(lib_, mask, window, max_vals)
        idx_arr = _full_like_int(lib_, max_vals, flat_idx)
        max_idx = _where(lib_, mask, idx_arr, max_idx)

    return max_vals, max_idx


def _pool_backward_avg(
    lib_: ModuleType,
    grad_out: _Array,
    input_shape: _Shape,
    out_dims: _Shape,
    kernel_size: _Shape,
    stride: _Shape,
    padding: _Shape,
) -> _Array:
    pad_shape = list(input_shape)
    for i, p in enumerate(padding):
        pad_shape[2 + i] += 2 * p

    grad_input_pad = _zeros(lib_, tuple(pad_shape), dtype=grad_out.dtype)
    grad_scaled = grad_out / _prod(kernel_size)

    for k_idx in itertools.product(*[range(k) for k in kernel_size]):
        slices = [slice(None), slice(None)]
        for d in range(len(kernel_size)):
            start = k_idx[d]
            end = start + stride[d] * out_dims[d]
            slices.append(slice(start, end, stride[d]))

        if lib_ is np:
            grad_input_pad[tuple(slices)] += grad_scaled
        else:
            grad_input_pad = grad_input_pad.at[tuple(slices)].add(grad_scaled)

    return _crop_padding(grad_input_pad, padding)


def _pool_backward_max(
    lib_: ModuleType,
    grad_out: _Array,
    input_shape: _Shape,
    out_dims: _Shape,
    kernel_size: _Shape,
    stride: _Shape,
    padding: _Shape,
    max_idx: _Array,
) -> _Array:
    pad_shape = list(input_shape)
    for i, p in enumerate(padding):
        pad_shape[2 + i] += 2 * p

    grad_input_pad = _zeros(lib_, tuple(pad_shape), dtype=grad_out.dtype)

    for flat_idx, k_idx in enumerate(
        itertools.product(*[range(k) for k in kernel_size])
    ):
        slices = [slice(None), slice(None)]
        for d in range(len(kernel_size)):
            start = k_idx[d]
            end = start + stride[d] * out_dims[d]
            slices.append(slice(start, end, stride[d]))

        mask = max_idx == flat_idx
        grad_contrib = grad_out * mask
        if lib_ is np:
            grad_input_pad[tuple(slices)] += grad_contrib
        else:
            grad_input_pad = grad_input_pad.at[tuple(slices)].add(grad_contrib)

    return _crop_padding(grad_input_pad, padding)


class pool_nd_kernel(Operation):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...] | list[int],
        stride: int | tuple[int, ...] | list[int],
        padding: int | tuple[int, ...] | list[int],
        mode: _Mode,
    ) -> None:
        super().__init__()
        if mode not in {"avg", "max"}:
            raise ValueError(f"Unsupported pooling mode: {mode}")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode

        self._kernel_size: _Shape | None = None
        self._stride: _Shape | None = None
        self._padding: _Shape | None = None
        self._out_dims: _Shape | None = None
        self._input_shape: _Shape | None = None
        self._max_idx: _Array | None = None

    def clear(self) -> None:
        self.result = None
        self._kernel_size = None
        self._stride = None
        self._padding = None
        self._out_dims = None
        self._input_shape = None
        self._max_idx = None

    def _normalize(self, input_: Tensor) -> tuple[_Shape, _Shape, _Shape]:
        if input_.ndim < 3:
            raise ValueError("Input must have at least 3 dimensions (N, C, ...).")

        D = input_.ndim - 2
        kernel = _to_tuple(self.kernel_size, D, "kernel_size")
        stride = _to_tuple(self.stride, D, "stride")
        padding = _to_tuple(self.padding, D, "padding")

        self._kernel_size = kernel
        self._stride = stride
        self._padding = padding

        return kernel, stride, padding

    @unary_func_op()
    def cpu(self, a: Tensor) -> _FuncOpReturnType:
        kernel, stride, padding = self._normalize(a)
        out_dims = _pool_out_dims(a.shape[2:], kernel, stride, padding)
        self._out_dims = out_dims
        self._input_shape = a.shape

        x_pad = _pad_input(np, a.data, padding)
        if self.mode == "avg":
            out_sum = _pool_forward_sum(x_pad, out_dims, kernel, stride)
            out = out_sum / _prod(kernel)
        else:
            out, max_idx = _pool_forward_max(np, x_pad, out_dims, kernel, stride)
            self._max_idx = max_idx

        self.result = Tensor(out)
        return self.result, partial(self.__grad__, lib_=np)

    @unary_func_op(device="gpu")
    def gpu(self, a: Tensor) -> _FuncOpReturnType:
        kernel, stride, padding = self._normalize(a)
        out_dims = _pool_out_dims(a.shape[2:], kernel, stride, padding)
        self._out_dims = out_dims
        self._input_shape = a.shape

        x_pad = _pad_input(mx, a.data, padding)
        if self.mode == "avg":
            out_sum = _pool_forward_sum(x_pad, out_dims, kernel, stride)
            out = out_sum / _prod(kernel)
        else:
            out, max_idx = _pool_forward_max(mx, x_pad, out_dims, kernel, stride)
            self._max_idx = max_idx

        self.result = Tensor(out)
        return self.result, partial(self.__grad__, lib_=mx)

    def __grad__(self, lib_: ModuleType) -> _GradType:
        if (
            self._kernel_size is None
            or self._stride is None
            or self._padding is None
            or self._out_dims is None
            or self._input_shape is None
        ):
            raise RuntimeError("pool_nd backward called before forward.")

        grad_out = self.result.grad
        if self.mode == "avg":
            grad_input = _pool_backward_avg(
                lib_,
                grad_out,
                self._input_shape,
                self._out_dims,
                self._kernel_size,
                self._stride,
                self._padding,
            )
            return grad_input

        if self._max_idx is None:
            raise RuntimeError("max pool backward missing max indices.")

        grad_input = _pool_backward_max(
            lib_,
            grad_out,
            self._input_shape,
            self._out_dims,
            self._kernel_size,
            self._stride,
            self._padding,
            self._max_idx,
        )
        return grad_input

    def __flops__(self, a: Tensor) -> int:
        if self._kernel_size is None or self._out_dims is None:
            kernel, stride, padding = self._normalize(a)
            out_dims = _pool_out_dims(a.shape[2:], kernel, stride, padding)
        else:
            kernel = self._kernel_size
            out_dims = self._out_dims

        kernel_elems = _prod(kernel)
        out_elems = int(a.shape[0]) * int(a.shape[1]) * _prod(out_dims)

        if kernel_elems <= 0 or out_elems <= 0:
            return 0

        if self.mode == "avg":
            return out_elems * kernel_elems
        return out_elems * max(kernel_elems - 1, 0)
