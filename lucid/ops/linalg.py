"""
lucid.ops.linalg — linear algebra (mirrors `lucid_legacy/linalg/`).

GPU-only by design (linalg ops route through MLX's CPU stream internally).
For CPU-resident tensors, callers should `.gpu()` first.
"""

from __future__ import annotations

from typing import Sequence

from lucid._C.engine import linalg as _C_linalg
from lucid._tensor import Tensor
from lucid._bridge import impl_of


__all__ = [
    "inv", "det", "solve", "cholesky", "norm",
    "qr", "svd", "matrix_power", "pinv", "eig",
]


def inv(a: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_linalg.inv(impl_of(a)))


def det(a: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_linalg.det(impl_of(a)))


def solve(a: Tensor, b: Tensor, /) -> Tensor:
    return Tensor._wrap(_C_linalg.solve(impl_of(a), impl_of(b)))


def cholesky(a: Tensor, /, upper: bool = False) -> Tensor:
    return Tensor._wrap(_C_linalg.cholesky(impl_of(a), bool(upper)))


def norm(
    a: Tensor, /,
    ord: float = 2,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    if axis is None:
        ax_list: list[int] = []
    elif isinstance(axis, int):
        ax_list = [int(axis)]
    else:
        ax_list = [int(a) for a in axis]
    return Tensor._wrap(_C_linalg.norm(
        impl_of(a), float(ord), ax_list, bool(keepdims)))


def qr(a: Tensor, /) -> tuple[Tensor, Tensor]:
    Q, R = _C_linalg.qr(impl_of(a))
    return Tensor._wrap(Q), Tensor._wrap(R)


def svd(a: Tensor, /, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    pieces = _C_linalg.svd(impl_of(a), True)
    return tuple(Tensor._wrap(p) for p in pieces)


def matrix_power(a: Tensor, /, n: int) -> Tensor:
    return Tensor._wrap(_C_linalg.matrix_power(impl_of(a), int(n)))


def pinv(a: Tensor, /, rcond: float = 1e-12) -> Tensor:
    # rcond is accepted for API compatibility; engine pinv uses MLX defaults.
    return Tensor._wrap(_C_linalg.pinv(impl_of(a)))


def eig(a: Tensor, /, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    w, v = _C_linalg.eig(impl_of(a))
    return Tensor._wrap(w), Tensor._wrap(v)
