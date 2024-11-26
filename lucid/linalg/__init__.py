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


def eig(a: Tensor) -> tuple[Tensor, Tensor]:
    return _func.eig(a)


def qr(a: Tensor) -> tuple[Tensor, Tensor]:
    return _func.qr(a)


def svd(a: Tensor, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    return _func.svd(a, full_matrices)


def matrix_power(a: Tensor, n: int) -> Tensor:
    return _func.matrix_power(a, n)


def pinv(a: Tensor) -> Tensor:
    return _func.pinv(a)
