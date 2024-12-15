import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLike, _Scalar, _base_dtype


def zeros(
    shape: _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.zeros(shape), requires_grad, keep_grad, dtype)


def zeros_like(
    a: Tensor | _ArrayLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if isinstance(a, Tensor):
        a = a.data
    return Tensor(np.zeros_like(a), requires_grad, keep_grad, dtype)


def ones(
    shape: _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.ones(shape), requires_grad, keep_grad, dtype)


def ones_like(
    a: Tensor | _ArrayLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if isinstance(a, Tensor):
        a = a.data
    return Tensor(np.ones_like(a), requires_grad, keep_grad, dtype)


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.eye(N, M, k), requires_grad, keep_grad, dtype)


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    if not isinstance(v, Tensor):
        v = Tensor(v, requires_grad, keep_grad, dtype)
    return Tensor(np.diag(v.data, k), v.requires_grad, v.keep_grad)


def arange(
    start: _Scalar,
    stop: _Scalar,
    step: _Scalar,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.arange(start, stop, step), requires_grad, keep_grad, dtype)


def empty(
    shape: int | _ShapeLike,
    dtype: type = _base_dtype,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.empty(shape), requires_grad, keep_grad, dtype)


def empty_like(
    a: Tensor | _ArrayLike,
    dtype: type | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if isinstance(a, Tensor):
        a = a.data
    return Tensor(np.empty_like(a), requires_grad, keep_grad, dtype)
