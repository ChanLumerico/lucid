"""
`lucid.linalg`
--------------
Lucid's linear algebra package.
"""

from lucid._tensor import Tensor
from lucid.linalg import _func


def inv(a: Tensor) -> Tensor:
    return _func.inv(a)


def det(a: Tensor) -> Tensor:
    return _func.det(a)


def solve(a: Tensor, b: Tensor) -> Tensor:
    return _func.solve(a, b)


def cholesky(a: Tensor) -> Tensor:
    return _func.cholesky(a)


def norm(a: Tensor, ord: int = 2) -> Tensor:
    return _func.norm(a, ord)
