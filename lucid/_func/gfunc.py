import numpy as np

from lucid._tensor import Tensor
from lucid import types
from lucid.types import (
    _ShapeLike,
    _ArrayLike,
    _Scalar,
    _DeviceType,
    _NumPyArray,
    _MLXArray,
    _BuiltinNumeric,
    Numeric,
)


def _get_tensor_specs(
    a: Tensor | _ArrayLike,
    dtype: _BuiltinNumeric | Numeric | None,
    device: _DeviceType | None,
) -> tuple[_ArrayLike, type, _DeviceType]:
    if isinstance(a, Tensor):
        data = a.data
        dtype = a.dtype if dtype is None else dtype
        device = a.device if device is None else device
    else:
        data = a
        dtype = types.to_numeric_type(a.dtype)
        if isinstance(a, _NumPyArray):
            device = "cpu" if device is None else device
        elif isinstance(a, _MLXArray):
            device = "gpu" if device is None else device
        else:
            raise ValueError(f"Unknown input: '{type(a).__name__}'")

    return data, dtype, device


def zeros(
    shape: _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad, keep_grad, dtype, device)


def zeros_like(
    a: Tensor | _ArrayLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    data, dtype, device = _get_tensor_specs(a, dtype, device)
    return Tensor(np.zeros_like(data), requires_grad, keep_grad, dtype, device)


def ones(
    shape: _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.ones(shape), requires_grad, keep_grad, dtype, device)


def ones_like(
    a: Tensor | _ArrayLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    data, dtype, device = _get_tensor_specs(a, dtype, device)
    return Tensor(np.ones_like(data), requires_grad, keep_grad, dtype, device)


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.eye(N, M, k), requires_grad, keep_grad, dtype, device)


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    data, dtype, device = _get_tensor_specs(v, dtype, device)
    return Tensor(np.diag(data, k=k), requires_grad, keep_grad, dtype, device)


def arange(
    start: _Scalar,
    stop: _Scalar,
    step: _Scalar,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.arange(start, stop, step), requires_grad, keep_grad, dtype, device)


def empty(
    shape: int | _ShapeLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.empty(shape), requires_grad, keep_grad, dtype, device)


def empty_like(
    a: Tensor | _ArrayLike,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    data, dtype, device = _get_tensor_specs(a, dtype, device)
    return Tensor(np.empty_like(data), requires_grad, keep_grad, dtype, device)


def linspace(
    start: _Scalar,
    stop: _Scalar,
    num: int = 50,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(
        np.linspace(start, stop, num), requires_grad, keep_grad, dtype, device
    )


def full(
    shape: int | _ShapeLike,
    fill_value: _Scalar,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.full(shape, fill_value), requires_grad, keep_grad, dtype, device)


def full_like(
    a: Tensor | _ArrayLike,
    fill_value: _Scalar,
    dtype: _BuiltinNumeric | Numeric | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    data, dtype, device = _get_tensor_specs(a, dtype, device)
    return Tensor(
        np.full_like(data, fill_value), requires_grad, keep_grad, dtype, device
    )
