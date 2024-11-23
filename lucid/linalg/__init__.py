from lucid._tensor import Tensor
from lucid.linalg import func


def inv(a: Tensor) -> Tensor:
    return func.inv(a)


def det(a: Tensor) -> Tensor:
    return func.det(a)


def solve(a: Tensor, b: Tensor) -> Tensor:
    return func.solve(a, b)


def cholesky(a: Tensor) -> Tensor:
    return func.cholesky(a)


def norm(a: Tensor, ord: int = 2) -> Tensor:
    return func.norm(a, ord)


def eig(a: Tensor) -> tuple[Tensor, Tensor]:
    return func.eig(a)


def qr(a: Tensor) -> tuple[Tensor, Tensor]:
    return func.qr(a)


def svd(a: Tensor, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    return func.svd(a, full_matrices)


def matrix_power(a: Tensor, n: int) -> Tensor:
    return func.matrix_power(a, n)


def pinv(a: Tensor) -> Tensor:
    return func.pinv(a)
