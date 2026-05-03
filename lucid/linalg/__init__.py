"""
lucid.linalg: linear algebra operations.
"""

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_la = _C_engine.linalg


def inv(x: Tensor) -> Tensor:
    """Matrix inverse."""
    return _wrap(_la.inv(_unwrap(x)))


def det(x: Tensor) -> Tensor:
    """Matrix determinant."""
    return _wrap(_la.det(_unwrap(x)))


def solve(A: Tensor, b: Tensor) -> Tensor:
    """Solve linear system Ax = b."""
    return _wrap(_la.solve(_unwrap(A), _unwrap(b)))


def cholesky(x: Tensor) -> Tensor:
    """Cholesky decomposition."""
    return _wrap(_la.cholesky(_unwrap(x)))


def norm(
    x: Tensor,
    ord: int | float | str | None = None,
    dim: int | list[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Matrix or vector norm."""
    return _wrap(_la.norm(_unwrap(x)))


def qr(x: Tensor) -> tuple[Tensor, Tensor]:
    """QR decomposition."""
    q, r = _la.qr(_unwrap(x))
    return _wrap(q), _wrap(r)


def svd(
    x: Tensor, full_matrices: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    """Singular value decomposition."""
    u, s, v = _la.svd(_unwrap(x))
    return _wrap(u), _wrap(s), _wrap(v)


def matrix_power(x: Tensor, n: int) -> Tensor:
    """Raise a matrix to an integer power."""
    return _wrap(_la.matrix_power(_unwrap(x), n))


def pinv(x: Tensor) -> Tensor:
    """Moore-Penrose pseudo-inverse."""
    return _wrap(_la.pinv(_unwrap(x)))


def eig(x: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalue decomposition."""
    vals, vecs = _la.eig(_unwrap(x))
    return _wrap(vals), _wrap(vecs)


def eigh(x: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalue decomposition of a symmetric matrix."""
    vals, vecs = _la.eigh(_unwrap(x))
    return _wrap(vals), _wrap(vecs)


__all__ = [
    "inv", "det", "solve", "cholesky", "norm", "qr", "svd",
    "matrix_power", "pinv", "eig", "eigh",
]
