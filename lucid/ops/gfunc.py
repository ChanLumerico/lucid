"""
lucid.ops.gfunc — tensor generation (mirrors `lucid_legacy/_func/gfunc.py`).

Includes: zeros / ones / full / empty / eye / arange / linspace / diag,
plus the `_like` family.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._bridge import (
    impl_of,
    normalize_shape,
    to_engine_dtype,
    to_engine_device,
)
from lucid.types import (
    Numeric,
    _ArrayLike,
    _BuiltinNumeric,
    _DeviceType,
    _Scalar,
    _ShapeLike,
)


__all__ = [
    "zeros", "ones", "full", "empty", "eye", "arange", "linspace", "diag",
    "zeros_like", "ones_like", "empty_like", "full_like",
]


# --------------------------------------------------------------------------- #
# Shape-explicit creation
# --------------------------------------------------------------------------- #

def zeros(
    *args: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    shape = normalize_shape(args)
    return Tensor._wrap(_C_engine.zeros(
        shape, to_engine_dtype(dtype), to_engine_device(device), requires_grad
    ))


def ones(
    *args: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    shape = normalize_shape(args)
    return Tensor._wrap(_C_engine.ones(
        shape, to_engine_dtype(dtype), to_engine_device(device), requires_grad
    ))


def full(
    shape: int | _ShapeLike,
    fill_value: _Scalar,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    sh = normalize_shape(shape if isinstance(shape, (list, tuple)) else (shape,))
    return Tensor._wrap(_C_engine.full(
        sh, float(fill_value), to_engine_dtype(dtype),
        to_engine_device(device), requires_grad
    ))


def empty(
    *args: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    shape = normalize_shape(args)
    return Tensor._wrap(_C_engine.empty(
        shape, to_engine_dtype(dtype), to_engine_device(device), requires_grad
    ))


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor._wrap(_C_engine.eye(
        int(N), int(M) if M is not None else -1, int(k),
        to_engine_dtype(dtype), to_engine_device(device), requires_grad
    ))


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    if not isinstance(v, Tensor):
        v = Tensor(v, dtype=dtype, device=device or "cpu",
                   requires_grad=requires_grad)
    return Tensor._wrap(_C_engine.diag(impl_of(v), int(k)))


def arange(
    *args,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise ValueError(f"arange expects 1-3 positional args, got {len(args)}")
    return Tensor._wrap(_C_engine.arange(
        float(start), float(stop), float(step),
        to_engine_dtype(dtype), to_engine_device(device), requires_grad
    ))


def linspace(
    start: _Scalar,
    stop: _Scalar,
    num: int = 50,
    /,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor._wrap(_C_engine.linspace(
        float(start), float(stop), int(num),
        to_engine_dtype(dtype), to_engine_device(device), requires_grad
    ))


# --------------------------------------------------------------------------- #
# `_like` family — copy shape/dtype/device of an existing tensor
# --------------------------------------------------------------------------- #

def zeros_like(
    a: Tensor | _ArrayLike,
    /,
    dtype=None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if dtype is None and device is None:
        return Tensor._wrap(_C_engine.zeros_like(impl_of(a), requires_grad))
    return zeros(*a.shape, dtype=dtype or a.dtype,
                 device=device or a.device, requires_grad=requires_grad)


def ones_like(
    a: Tensor | _ArrayLike,
    /,
    dtype=None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if dtype is None and device is None:
        return Tensor._wrap(_C_engine.ones_like(impl_of(a), requires_grad))
    return ones(*a.shape, dtype=dtype or a.dtype,
                device=device or a.device, requires_grad=requires_grad)


def empty_like(
    a: Tensor | _ArrayLike,
    /,
    dtype=None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if dtype is None and device is None:
        return Tensor._wrap(_C_engine.empty_like(impl_of(a), requires_grad))
    return empty(*a.shape, dtype=dtype or a.dtype,
                 device=device or a.device, requires_grad=requires_grad)


def full_like(
    a: Tensor | _ArrayLike,
    fill_value: _Scalar,
    dtype=None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if dtype is None and device is None:
        return Tensor._wrap(_C_engine.full_like(
            impl_of(a), float(fill_value), requires_grad))
    return full(a.shape, fill_value,
                dtype=dtype or a.dtype, device=device or a.device,
                requires_grad=requires_grad)
