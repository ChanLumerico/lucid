"""
Tensor creation functions: zeros, ones, empty, full, eye, arange, linspace, *_like.
"""

from typing import Any, TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs, _unwrap
from lucid._dtype import dtype

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _size_to_list(*size: Any) -> list[int]:
    """Normalize size args: zeros(2,3) or zeros((2,3)) → [2, 3]."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        return list(size[0])
    return list(size)


def zeros(
    *size: Any,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
    requires_grad: bool = False,
) -> "Tensor":
    """Return a tensor filled with zeros."""
    from lucid._dispatch import _wrap
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    return _wrap(_C_engine.zeros(shape, _dt, _dev))


def ones(
    *size: Any,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
    requires_grad: bool = False,
) -> "Tensor":
    """Return a tensor filled with ones."""
    from lucid._dispatch import _wrap
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    return _wrap(_C_engine.ones(shape, _dt, _dev))


def empty(
    *size: Any,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
    requires_grad: bool = False,
) -> "Tensor":
    """Return an uninitialized tensor."""
    from lucid._dispatch import _wrap
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    return _wrap(_C_engine.empty(shape, _dt, _dev))


def full(
    size: "int | list[int] | tuple[int, ...]",
    fill_value: float,
    *,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
    requires_grad: bool = False,
) -> "Tensor":
    """Return a tensor filled with fill_value."""
    from lucid._dispatch import _wrap
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = list(size) if isinstance(size, (list, tuple)) else [size]
    return _wrap(_C_engine.full(shape, fill_value, _dt, _dev))


def eye(
    n: int,
    m: int | None = None,
    *,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
) -> "Tensor":
    """Return a 2D identity matrix."""
    from lucid._dispatch import _wrap
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    _m = m if m is not None else n
    return _wrap(_C_engine.eye(n, _m, _dt, _dev))


def arange(
    start: float,
    end: float | None = None,
    step: float = 1,
    *,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
) -> "Tensor":
    """Return evenly spaced values within a given interval."""
    from lucid._dispatch import _wrap
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    if end is None:
        start, end = 0.0, float(start)
    return _wrap(_C_engine.arange(start, end, step, _dt, _dev))


def linspace(
    start: float,
    end: float,
    steps: int,
    *,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
) -> "Tensor":
    """Return evenly spaced numbers over a specified interval."""
    from lucid._dispatch import _wrap
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    return _wrap(_C_engine.linspace(start, end, steps, _dt, _dev))


def zeros_like(
    t: "Tensor",
    *,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
) -> "Tensor":
    """Return a zeros tensor with the same shape/dtype/device as t."""
    from lucid._dispatch import _wrap
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
    )
    return _wrap(_C_engine.zeros_like(_unwrap(t), _dt, _dev))


def ones_like(
    t: "Tensor",
    *,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
) -> "Tensor":
    """Return a ones tensor with the same shape/dtype/device as t."""
    from lucid._dispatch import _wrap
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
    )
    return _wrap(_C_engine.ones_like(_unwrap(t), _dt, _dev))


def empty_like(
    t: "Tensor",
    *,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
) -> "Tensor":
    """Return an uninitialized tensor with the same shape/dtype/device as t."""
    from lucid._dispatch import _wrap
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
    )
    return _wrap(_C_engine.empty_like(_unwrap(t), _dt, _dev))


def full_like(
    t: "Tensor",
    fill_value: float,
    *,
    dtype: "dtype | _C_engine.Dtype | str | None" = None,
    device: "str | None" = None,
) -> "Tensor":
    """Return a tensor filled with fill_value, shaped like t."""
    from lucid._dispatch import _wrap
    _dt, _dev, _ = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
    )
    return _wrap(_C_engine.full_like(_unwrap(t), fill_value, _dt, _dev))
