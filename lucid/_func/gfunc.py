from typing import Any
import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLike, _Scalar


def zeros(
    shape: _ShapeLike,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad, keep_grad)


def zeros_like(
    a: Tensor | _ArrayLike,
    dtype: Any = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if isinstance(a, Tensor):
        a = a.data
    return Tensor(np.zeros_like(a, dtype=dtype), requires_grad, keep_grad)


def ones(
    shape: _ShapeLike,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.ones(shape, dtype=dtype), requires_grad, keep_grad)


def ones_like(
    a: Tensor | _ArrayLike,
    dtype: Any = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if isinstance(a, Tensor):
        a = a.data
    return Tensor(np.ones_like(a, dtype=dtype), requires_grad, keep_grad)


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.eye(N, M, k, dtype=dtype), requires_grad, keep_grad)


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    dtype: Any = np.float32,
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
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    return Tensor(np.arange(start, stop, step), requires_grad, keep_grad, dtype)
