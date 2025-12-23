from functools import partial
from types import ModuleType
from typing import TypeAlias
import itertools
import os

import numpy as np

from lucid._tensor import Tensor
from lucid._backend.core import (
    operation,
    binary_func_op,
    _FuncOpReturnType,
    _GradFuncType,
)
from lucid._backend.metal import mx

from lucid.types import _NumPyArray, _MLXArray


_Array: TypeAlias = _NumPyArray | _MLXArray
_Shape: TypeAlias = tuple[int, ...]
_Stride: TypeAlias = tuple[int, ...]
_Padding: TypeAlias = tuple[int, ...]
_Dilation: TypeAlias = tuple[int, ...]


def _load_view_limit_bytes() -> int:
    env = os.getenv("LUCID_CONV_VIEW_LIMIT_MB")
    if env is None:
        return _default_view_limit_bytes()
    try:
        value = int(env)
    except ValueError:
        return _default_view_limit_bytes()
    return value * 1024 * 1024


def _sysconf_value(name: str) -> int | None:
    try:
        value = int(os.sysconf(name))
    except (ValueError, AttributeError, OSError):
        return None
    if value <= 0:
        return None
    return value


def _get_total_memory_bytes() -> int | None:
    page_size = _sysconf_value("SC_PAGE_SIZE") or _sysconf_value("SC_PAGESIZE")
    phys_pages = _sysconf_value("SC_PHYS_PAGES")
    if page_size and phys_pages:
        return page_size * phys_pages
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys)

    except Exception:
        return None


def _round_to_step(value: int, step: int) -> int:
    return ((value + step // 2) // step) * step


def _default_view_limit_bytes() -> int:
    total = _get_total_memory_bytes()
    if not total:
        return 256 * 1024 * 1024

    mb = 1024 * 1024
    min_bytes = 64 * mb
    max_bytes = 1024 * mb
    step = 64 * mb

    target = (total * 15) // 1000
    target = max(min_bytes, min(max_bytes, target))
    target = _round_to_step(target, step)
    return max(min_bytes, min(max_bytes, target))


_CONV_VIEW_LIMIT_BYTES = _load_view_limit_bytes()


def get_conv_view_limit_mb() -> int:
    return int(_CONV_VIEW_LIMIT_BYTES // (1024 * 1024))


def _dtype_itemsize(data: _Array) -> int:
    dtype = getattr(data, "dtype", None)
    if dtype is None:
        return 0
    try:
        return int(np.dtype(dtype).itemsize)
    except TypeError:
        return int(getattr(dtype, "size", 0) or 0)


def _prod(shape: _Shape) -> int:
    total = 1
    for v in shape:
        total *= int(v)
    return total


def _view_exceeds_limit(data: _Array, out_dims: _Shape, kernel_size: _Shape) -> bool:
    if _CONV_VIEW_LIMIT_BYTES == 0:
        return True
    if _CONV_VIEW_LIMIT_BYTES < 0:
        return False
    itemsize = _dtype_itemsize(data)
    if itemsize == 0:
        return False

    view_elems = data.shape[0] * data.shape[1] * _prod(out_dims) * _prod(kernel_size)
    view_bytes = view_elems * itemsize

    return view_bytes > _CONV_VIEW_LIMIT_BYTES


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


def _conv_out_dims(
    input_spatial: _Shape,
    kernel_size: _Shape,
    stride: _Stride,
    padding: _Padding,
    dilation: _Dilation,
) -> list[int]:
    out_dims = []
    for i in range(len(kernel_size)):
        eff = dilation[i] * (kernel_size[i] - 1) + 1
        o = (input_spatial[i] + 2 * padding[i] - eff) // stride[i] + 1
        if o <= 0:
            raise ValueError(f"Non-positive output dim for axis {i}: {o}")
        out_dims.append(o)

    return out_dims


def _validate_conv_shapes(input_: Tensor, weight: Tensor, groups: int) -> None:
    if input_.ndim != weight.ndim:
        raise ValueError("Input and weight must have the same number of dimensions.")
    if input_.ndim < 3:
        raise ValueError("Input and weight must have at least 3 dimensions.")
    if groups <= 0:
        raise ValueError("groups must be a positive integer.")

    C_in = input_.shape[1]
    C_out = weight.shape[0]
    C_in_g = weight.shape[1]

    if C_out % groups != 0 or C_in_g * groups != C_in:
        raise ValueError("Inconsistent channel/group configuration.")


def _pad_input(lib_: ModuleType, data: _Array, padding: _Padding) -> _Array:
    if not any(padding):
        return data

    pad_width = ((0, 0), (0, 0)) + tuple((p, p) for p in padding)
    return lib_.pad(data, pad_width)


def _as_strided(
    lib_: ModuleType, data: _Array, shape: _Shape, strides: _Shape
) -> _Array | None:
    if lib_ is np:
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    as_strided = getattr(lib_, "as_strided", None)
    if as_strided is None:
        return None

    try:
        return as_strided(data, shape=shape, strides=strides)
    except TypeError:
        return as_strided(data, shape, strides)


def _make_input_view(
    lib_: ModuleType,
    data: _Array,
    out_dims: _Shape,
    kernel_size: _Shape,
    stride: _Stride,
    dilation: _Dilation,
) -> _Array | None:
    if not hasattr(data, "strides"):
        return None
    strides = data.strides
    if strides is None:
        return None

    spatial_strides = strides[2:]
    view_strides = (
        strides[0],
        strides[1],
        *[spatial_strides[i] * stride[i] for i in range(len(kernel_size))],
        *[spatial_strides[i] * dilation[i] for i in range(len(kernel_size))],
    )
    view_shape = (data.shape[0], data.shape[1], *out_dims, *kernel_size)

    return _as_strided(lib_, data, view_shape, view_strides)


def _conv_from_view(
    lib_: ModuleType, x_view: _Array, weight: _Array, out_dims: _Shape, groups: int
) -> _Array:
    D = len(out_dims)
    C_out = weight.shape[0]
    C_in_g = weight.shape[1]
    C_out_g = C_out // groups

    axes_x = [1] + list(range(2 + D, 2 + 2 * D))
    axes_w = [1] + list(range(2, 2 + D))
    perm = [0, D + 1] + list(range(1, D + 1))

    outputs = []
    for g in range(groups):
        x_g = x_view[:, g * C_in_g : (g + 1) * C_in_g, ...]
        w_g = weight[g * C_out_g : (g + 1) * C_out_g, ...]

        out = lib_.tensordot(x_g, w_g, axes=(axes_x, axes_w))
        out = lib_.transpose(out, axes=perm)
        outputs.append(out)

    if len(outputs) == 1:
        return outputs[0]

    return lib_.concatenate(outputs, axis=1)


def _conv_fallback(
    lib_: ModuleType,
    input_: _Array,
    weight: _Array,
    stride: _Stride,
    padding: _Padding,
    dilation: _Dilation,
    groups: int,
    out_dims: _Shape,
) -> _Array:
    D = len(out_dims)
    kernel_size = weight.shape[2:]
    C_out = weight.shape[0]
    C_in_g = weight.shape[1]
    C_out_g = C_out // groups

    x = _pad_input(lib_, input_, padding)

    outputs = []
    for g in range(groups):
        x_g = x[:, g * C_in_g : (g + 1) * C_in_g]
        w_g = weight[g * C_out_g : (g + 1) * C_out_g]

        out_g = None
        for k_idx in itertools.product(*[range(k) for k in kernel_size]):
            slices = [slice(None), slice(None)]

            for d in range(D):
                start = k_idx[d] * dilation[d]
                end = start + stride[d] * out_dims[d]
                slices.append(slice(start, end, stride[d]))

            x_slice = x_g[tuple(slices)]
            w_slice = w_g[(slice(None), slice(None)) + k_idx]

            contrib = lib_.tensordot(x_slice, w_slice, axes=([1], [1]))
            perm = [0, contrib.ndim - 1] + list(range(1, contrib.ndim - 1))

            contrib = lib_.transpose(contrib, axes=perm)
            out_g = contrib if out_g is None else out_g + contrib

        outputs.append(out_g)

    if len(outputs) == 1:
        return outputs[0]

    return lib_.concatenate(outputs, axis=1)


def _conv_forward(
    lib_: ModuleType,
    input_: _Array,
    weight: _Array,
    stride: _Stride,
    padding: _Padding,
    dilation: _Dilation,
    groups: int,
) -> _Array:
    input_spatial = input_.shape[2:]
    kernel_size = weight.shape[2:]
    out_dims = tuple(
        _conv_out_dims(input_spatial, kernel_size, stride, padding, dilation)
    )

    if _view_exceeds_limit(input_, out_dims, kernel_size):
        return _conv_fallback(
            lib_, input_, weight, stride, padding, dilation, groups, out_dims
        )

    x = _pad_input(lib_, input_, padding)
    x_view = _make_input_view(lib_, x, out_dims, kernel_size, stride, dilation)
    if x_view is None:
        return _conv_fallback(
            lib_, input_, weight, stride, padding, dilation, groups, out_dims
        )

    return _conv_from_view(lib_, x_view, weight, out_dims, groups)


def _conv_backward_weight(
    lib_: ModuleType,
    grad_out: _Array,
    x_pad: _Array,
    weight: _Array,
    stride: _Stride,
    dilation: _Dilation,
    groups: int,
) -> _Array:
    weight_shape = weight.shape
    D = len(weight_shape) - 2
    out_dims = grad_out.shape[2:]
    kernel_size = weight.shape[2:]
    C_out = weight_shape[0]
    C_in_g = weight_shape[1]
    C_out_g = C_out // groups

    x_view = _make_input_view(lib_, x_pad, out_dims, kernel_size, stride, dilation)
    if x_view is not None and _view_exceeds_limit(x_pad, out_dims, kernel_size):
        x_view = None
    axes_out = [0] + list(range(2, 2 + D))
    axes_x = [0] + list(range(2, 2 + D))

    grad_parts = []
    for g in range(groups):
        grad_out_g = grad_out[:, g * C_out_g : (g + 1) * C_out_g, ...]

        if x_view is None:
            x_g = x_pad[:, g * C_in_g : (g + 1) * C_in_g]
            grad_w = lib_.zeros((C_out_g, C_in_g, *kernel_size), dtype=weight.dtype)

            for k_idx in itertools.product(*[range(k) for k in kernel_size]):
                slices = [slice(None), slice(None)]

                for d in range(D):
                    start = k_idx[d] * dilation[d]
                    end = start + stride[d] * out_dims[d]
                    slices.append(slice(start, end, stride[d]))

                x_slice = x_g[tuple(slices)]
                w_grad = lib_.tensordot(grad_out_g, x_slice, axes=(axes_out, axes_x))

                if lib_ is np:
                    grad_w[(slice(None), slice(None)) + k_idx] = w_grad
                else:
                    grad_w = grad_w.at[(slice(None), slice(None)) + k_idx].add(w_grad)
            grad_parts.append(grad_w)

        else:
            x_view_g = x_view[:, g * C_in_g : (g + 1) * C_in_g, ...]
            grad_w = lib_.tensordot(grad_out_g, x_view_g, axes=(axes_out, axes_x))
            grad_parts.append(grad_w)

    if len(grad_parts) == 1:
        return grad_parts[0]

    return lib_.concatenate(grad_parts, axis=0)


def _conv_backward_input(
    lib_: ModuleType,
    grad_out: _Array,
    weight: _Array,
    x_pad: _Array,
    stride: _Stride,
    padding: _Padding,
    dilation: _Dilation,
    groups: int,
) -> _Array:
    kernel_size = weight.shape[2:]
    D = len(kernel_size)
    out_dims = grad_out.shape[2:]

    C_out = weight.shape[0]
    C_in_g = weight.shape[1]
    C_out_g = C_out // groups

    grad_input = lib_.zeros_like(x_pad)

    for g in range(groups):
        grad_out_g = grad_out[:, g * C_out_g : (g + 1) * C_out_g, ...]
        w_g = weight[g * C_out_g : (g + 1) * C_out_g]
        ch_slice = slice(g * C_in_g, (g + 1) * C_in_g)

        for k_idx in itertools.product(*[range(k) for k in kernel_size]):
            w_slice = w_g[(slice(None), slice(None)) + k_idx]
            contrib = lib_.tensordot(grad_out_g, w_slice, axes=([1], [0]))

            perm = [0, contrib.ndim - 1] + list(range(1, contrib.ndim - 1))
            contrib = lib_.transpose(contrib, axes=perm)

            slices = [slice(None), ch_slice]
            for d in range(D):
                start = k_idx[d] * dilation[d]
                end = start + stride[d] * out_dims[d]
                slices.append(slice(start, end, stride[d]))

            if lib_ is np:
                grad_input[tuple(slices)] += contrib
            else:
                grad_input = grad_input.at[tuple(slices)].add(contrib)

    if any(padding):
        crop = [slice(None), slice(None)]
        for p in padding:
            end = -p if p != 0 else None
            crop.append(slice(p, end))
        return grad_input[tuple(crop)]

    return grad_input


class conv_nd(operation):
    def __init__(
        self,
        stride: int | tuple[int, ...] | list[int],
        padding: int | tuple[int, ...] | list[int],
        dilation: int | tuple[int, ...] | list[int],
        groups: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self._stride: _Stride | None = None
        self._padding: _Padding | None = None
        self._dilation: _Dilation | None = None

    def _normalize(self, weight: Tensor) -> tuple[_Stride, _Padding, _Dilation]:
        D = weight.ndim - 2
        stride = _to_tuple(self.stride, D, "stride")
        padding = _to_tuple(self.padding, D, "padding")
        dilation = _to_tuple(self.dilation, D, "dilation")

        self._stride = stride
        self._padding = padding
        self._dilation = dilation

        return stride, padding, dilation

    @binary_func_op()
    def cpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        _validate_conv_shapes(a, b, self.groups)
        stride, padding, dilation = self._normalize(b)
        out = _conv_forward(np, a.data, b.data, stride, padding, dilation, self.groups)

        self.result = Tensor(out)
        return self.result, partial(self.__grad__, a=a, b=b, lib_=np)

    @binary_func_op(device="gpu")
    def gpu(self, a: Tensor, b: Tensor) -> _FuncOpReturnType:
        _validate_conv_shapes(a, b, self.groups)
        stride, padding, dilation = self._normalize(b)
        out = _conv_forward(mx, a.data, b.data, stride, padding, dilation, self.groups)

        self.result = Tensor(out)
        return self.result, partial(self.__grad__, a=a, b=b, lib_=mx)

    def __grad__(self, a: Tensor, b: Tensor, lib_: ModuleType) -> _GradFuncType:
        stride = self._stride
        padding = self._padding
        dilation = self._dilation

        if stride is None or padding is None or dilation is None:
            raise RuntimeError("conv_nd backward called before forward.")

        x_pad = _pad_input(lib_, a.data, padding)
        grad_out = self.result.grad

        grad_input = _conv_backward_input(
            lib_, grad_out, b.data, x_pad, stride, padding, dilation, self.groups
        )
        grad_weight = _conv_backward_weight(
            lib_, grad_out, x_pad, b.data, stride, dilation, self.groups
        )

        return grad_input, grad_weight


def conv_nd_op(
    stride: int | tuple[int, ...] | list[int],
    padding: int | tuple[int, ...] | list[int],
    dilation: int | tuple[int, ...] | list[int],
    groups: int,
) -> conv_nd:
    return conv_nd(stride, padding, dilation, groups)
