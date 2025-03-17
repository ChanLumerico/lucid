import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLike, _Scalar, _DeviceType


def zeros(
    shape: _ShapeLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad, keep_grad, dtype, device)


def zeros_like(
    a: Tensor | _ArrayLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if device is None:
        device = a.device if hasattr(a, "device") else "cpu"
    if isinstance(a, Tensor):
        a = a.data

    return Tensor(np.zeros_like(a), requires_grad, keep_grad, dtype, device)


def ones(
    shape: _ShapeLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.ones(shape), requires_grad, keep_grad, dtype, device)


def ones_like(
    a: Tensor | _ArrayLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if device is None:
        device = a.device if hasattr(a, "device") else "cpu"
    if isinstance(a, Tensor):
        a = a.data

    return Tensor(np.ones_like(a), requires_grad, keep_grad, dtype, device)


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.eye(N, M, k), requires_grad, keep_grad, dtype, device)


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType | None = None,
) -> Tensor:
    if dtype is None and hasattr(v, "dtype"):
        dtype = v.dtype
    if device is None:
        device = v.device if hasattr(v, "device") else "cpu"
    if not isinstance(v, Tensor):
        v = Tensor(v)

    return Tensor(np.diag(v.data, k=k), requires_grad, keep_grad, dtype, device)


def arange(
    start: _Scalar,
    stop: _Scalar,
    step: _Scalar,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.arange(start, stop, step), requires_grad, keep_grad, dtype, device)


def empty(
    shape: int | _ShapeLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(np.empty(shape), requires_grad, keep_grad, dtype, device)


def empty_like(
    a: Tensor | _ArrayLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if isinstance(a, Tensor):
        a = a.data
    if device is None:
        device = a.device if hasattr(a, "device") else "cpu"

    return Tensor(np.empty_like(a), requires_grad, keep_grad, dtype, device)


def linspace(
    start: _Scalar,
    stop: _Scalar,
    num: int = 50,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
    device: _DeviceType = "cpu",
) -> Tensor:
    return Tensor(
        np.linspace(start, stop, num), requires_grad, keep_grad, dtype, device
    )
