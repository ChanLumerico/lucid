"""
Tensor creation functions: zeros, ones, empty, full, eye, arange, linspace, *_like.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs, _unwrap, _wrap, _impl_with_grad
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _size_to_list(*size: int | tuple[int, ...]) -> list[int]:
    """Normalize size args: zeros(2,3) or zeros((2,3)) → [2, 3]."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        return list(size[0])
    return list(size)  # type: ignore[arg-type]


def zeros(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with zeros.

    Parameters
    ----------
    *size : int
        Shape of the output tensor. Can be passed as separate ints or a tuple.
    dtype : lucid.dtype, optional
        Desired data type. Defaults to ``lucid.float32``.
    device : str, optional
        Target device (``"cpu"`` or ``"metal"``).
    requires_grad : bool, optional
        Enable gradient tracking.

    Returns
    -------
    Tensor
        Zero tensor of the given shape.

    Examples
    --------
    >>> lucid.zeros(2, 3).shape
    (2, 3)
    """
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    impl = _C_engine.zeros(shape, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def ones(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with ones.

    Parameters
    ----------
    *size : int
        Shape of the output tensor.
    dtype : lucid.dtype, optional
        Desired data type. Defaults to ``lucid.float32``.
    device : str, optional
        Target device.
    requires_grad : bool, optional
        Enable gradient tracking.

    Returns
    -------
    Tensor
        All-ones tensor of the given shape.
    """
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    impl = _C_engine.ones(shape, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def empty(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return an uninitialized tensor.

    The values are undefined (contents of uninitialised memory). Use only when
    you intend to fill the tensor before reading.

    Parameters
    ----------
    *size : int
        Shape of the output tensor.
    dtype : lucid.dtype, optional
        Desired data type.
    device : str, optional
        Target device.
    requires_grad : bool, optional
        Enable gradient tracking.

    Returns
    -------
    Tensor
        Uninitialized tensor of the given shape.
    """
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    impl = _C_engine.empty(shape, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def full(
    size: int | list[int] | tuple[int, ...],
    fill_value: float,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with fill_value."""
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = list(size) if isinstance(size, (list, tuple)) else [size]
    impl = _C_engine.full(shape, fill_value, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def eye(
    n: int,
    m: int | None = None,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a 2D identity matrix."""
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    _m = m if m is not None else n
    impl = _C_engine.eye(n, _m, 0, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def arange(
    start: float,
    end: float | None = None,
    step: float = 1,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Return evenly spaced values within a given interval."""
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    if end is None:
        start, end = 0.0, float(start)
    return _wrap(_C_engine.arange(start, end, step, _dt, _dev))


def linspace(
    start: float,
    end: float,
    steps: int,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Return evenly spaced numbers over a specified interval."""
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    return _wrap(_C_engine.linspace(start, end, steps, _dt, _dev))


def zeros_like(
    t: Tensor,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a zeros tensor with the same shape/dtype/device as t."""
    _dt, _dev, _rg = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
        requires_grad,
    )
    impl = _unwrap(t)
    out = _C_engine.zeros(list(impl.shape), _dt, _dev)
    return _wrap(_impl_with_grad(out, _rg) if _rg else out)


def ones_like(
    t: Tensor,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a ones tensor with the same shape/dtype/device as t."""
    _dt, _dev, _rg = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
        requires_grad,
    )
    impl = _unwrap(t)
    out = _C_engine.ones(list(impl.shape), _dt, _dev)
    return _wrap(_impl_with_grad(out, _rg) if _rg else out)


def empty_like(
    t: Tensor,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return an uninitialized tensor with the same shape/dtype/device as t."""
    _dt, _dev, _rg = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
        requires_grad,
    )
    impl = _unwrap(t)
    out = _C_engine.empty(list(impl.shape), _dt, _dev)
    return _wrap(_impl_with_grad(out, _rg) if _rg else out)


def full_like(
    t: Tensor,
    fill_value: float,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Return a tensor filled with fill_value, shaped like t.

    The engine's ``full_like`` always inherits dtype/device from ``t``; the
    ``dtype`` / ``device`` kwargs are accepted for source-level compatibility
    with the reference framework, but a non-default value triggers a cast +
    move via ``astype`` / ``to`` so the returned tensor matches.
    """
    out: Tensor = _wrap(_C_engine.full_like(_unwrap(t), fill_value, False))
    if dtype is not None and dtype is not t.dtype:
        out = out.astype(dtype)  # type: ignore[attr-defined]
    if device is not None and str(device) != str(t.device):
        out = out.to(device)
    return out


def logspace(
    start: float,
    end: float,
    steps: int,
    base: float = 10.0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Return *steps* values spaced evenly on a log scale (base^start … base^end)."""
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    return _wrap(_C_engine.logspace(start, end, steps, base, _dt, _dev))
