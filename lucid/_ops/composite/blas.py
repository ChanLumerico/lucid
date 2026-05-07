"""BLAS-level composites (``addmm``, ``addbmm``, ``mv``, ``ger``, ...)."""

from typing import TYPE_CHECKING

import lucid

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    """``β · input + α · (mat1 @ mat2)``."""
    return input * beta + lucid.matmul(mat1, mat2) * alpha


def addbmm(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    """``β · input + α · Σ_k (batch1[k] @ batch2[k])``."""
    return input * beta + lucid.sum(lucid.bmm(batch1, batch2), 0) * alpha


def baddbmm(
    input: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    """``β · input + α · bmm(batch1, batch2)``."""
    return input * beta + lucid.bmm(batch1, batch2) * alpha


def addmv(
    input: Tensor,
    mat: Tensor,
    vec: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    """``β · input + α · (mat @ vec)``."""
    mv_out = lucid.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)
    return input * beta + mv_out * alpha


def addr(
    input: Tensor,
    vec1: Tensor,
    vec2: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    """``β · input + α · outer(vec1, vec2)``."""
    out = lucid.matmul(vec1.unsqueeze(-1), vec2.unsqueeze(0))
    return input * beta + out * alpha


def addcmul(
    input: Tensor,
    t1: Tensor,
    t2: Tensor,
    *,
    value: float = 1.0,
) -> Tensor:
    return input + (t1 * t2) * value


def addcdiv(
    input: Tensor,
    t1: Tensor,
    t2: Tensor,
    *,
    value: float = 1.0,
) -> Tensor:
    return input + (t1 / t2) * value


def mv(mat: Tensor, vec: Tensor) -> Tensor:
    return lucid.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)


def ger(vec1: Tensor, vec2: Tensor) -> Tensor:
    """Outer product of two 1-D tensors — alias of ``outer``."""
    return lucid.linalg.outer(vec1, vec2)


def vdot(a: Tensor, b: Tensor) -> Tensor:
    """Real vector dot — alias of ``dot``."""
    return lucid.linalg.dot(a, b)


def block_diag(*tensors: Tensor) -> Tensor:
    """Block-diagonal matrix built from 0/1/2-D inputs."""
    if not tensors:
        return lucid.zeros(0, 0)

    parts: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.unsqueeze(0)
        parts.append(t_i)

    total_cols = sum(t_i.shape[1] for t_i in parts)
    rows: list[Tensor] = []
    cum = 0
    for blk in parts:
        h, w = blk.shape
        pieces: list[Tensor] = []
        if cum:
            pieces.append(lucid.zeros(h, cum, dtype=blk.dtype, device=blk.device))
        pieces.append(blk)
        right = total_cols - cum - w
        if right:
            pieces.append(lucid.zeros(h, right, dtype=blk.dtype, device=blk.device))
        rows.append(lucid.cat(pieces, 1))
        cum += w
    return lucid.cat(rows, 0)


__all__ = [
    "addmm",
    "addbmm",
    "baddbmm",
    "addmv",
    "addr",
    "addcmul",
    "addcdiv",
    "mv",
    "ger",
    "vdot",
    "block_diag",
]
