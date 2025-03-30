from lucid._tensor import Tensor
from lucid.linalg import _func


# fmt: off
__all__ = [
    "inv", "det", "solve", "cholesky", "norm", "qr", "svd", "matrix_power", "pinv"
]
# fmt: on


def inv(a: Tensor, /) -> Tensor:
    return _func.inv()(a)


def det(a: Tensor, /) -> Tensor:
    return _func.det()(a)


def solve(a: Tensor, b: Tensor, /) -> Tensor:
    return _func.solve()(a, b)


def cholesky(a: Tensor, /) -> Tensor:
    return _func.cholesky()(a)


def norm(
    a: Tensor,
    /,
    ord: int = 2,
    axis: tuple[int, ...] | int | None = None,
    keepdims: bool = False,
) -> Tensor:
    return _func.norm(ord, axis, keepdims)(a)


def eig(a: Tensor, /, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    return _func.eig(eps)(a)


def qr(a: Tensor, /) -> tuple[Tensor, Tensor]:
    return _func.qr()(a)


def svd(a: Tensor, /, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    return _func.svd(full_matrices)(a)


def matrix_power(a: Tensor, /, n: int) -> Tensor:
    return _func.matrix_power(n)(a)


def pinv(a: Tensor, /, rcond: float = 1e-12) -> Tensor:
    return _func.pinv(rcond)(a)
