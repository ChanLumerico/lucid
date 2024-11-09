from typing import Any
import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLike


__all__ = ["zeros", "zeros_like", "ones", "ones_like", "eye", "diag"]


def zeros(shape: _ShapeLike, dtype: Any = np.float32) -> Tensor:
    return Tensor(np.zeros(shape, dtype=dtype))


def zeros_like(a: Tensor | _ArrayLike, dtype: Any = None) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if isinstance(a, Tensor):
        a = a.data
    return Tensor(np.zeros_like(a, dtype=dtype))


def ones(shape: _ShapeLike, dtype: Any = np.float32) -> Tensor:
    return Tensor(np.ones(shape, dtype=dtype))


def ones_like(a: Tensor | _ArrayLike, dtype: Any = None) -> Tensor:
    if dtype is None and hasattr(a, "dtype"):
        dtype = a.dtype
    if isinstance(a, Tensor):
        a = a.data
    return Tensor(np.ones_like(a, dtype=dtype))


def eye(N: int, M: int | None = None, k: int = 0, dtype: Any = np.float32) -> Tensor:
    return Tensor(np.eye(N, M, k, dtype=dtype))


def diag(v: Tensor | _ArrayLike, k: int = 0) -> Tensor:
    if isinstance(v, Tensor):
        v = v.data
    return Tensor(np.diag(v, k))
