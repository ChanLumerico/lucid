"""
`lucid.linalg`
--------------
Lucid's linear algebra package.
"""

from lucid._tensor import Tensor
from lucid.linalg import _func


def inv(a: Tensor) -> Tensor:
    """Compute the inverse of a matrix."""
    return _func.inv(a)


def det(a: Tensor) -> Tensor:
    """Compute the determinant of a matrix."""
    return _func.det(a)


def solve(a: Tensor, b: Tensor) -> Tensor:
    """Solve a linear system `Ax=b`."""
    return _func.solve(a, b)


def cholesky(a: Tensor) -> Tensor:
    """Performs Cholesky decomposition."""
    return _func.cholesky(a)


def norm(a: Tensor, ord: int = 2) -> Tensor:
    """Compute the p-norm of a matrix."""
    return _func.norm(a, ord)
