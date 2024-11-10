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
