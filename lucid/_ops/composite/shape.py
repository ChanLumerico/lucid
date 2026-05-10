"""Shape-manipulation composites: axis swaps, stacks, splits, and the
miscellaneous fillers (``rot90``, ``vander``, ``take_along_dim``).
"""

from typing import Sequence, TYPE_CHECKING

import lucid
from lucid._types import DTypeLike, DeviceLike
from lucid._ops.composite._shared import _swap_dims

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Axis swaps ─────────────────────────────────────────────────────────────


def swapaxes(x: Tensor, axis0: int, axis1: int) -> Tensor:
    return _swap_dims(x, axis0, axis1)


def swapdims(x: Tensor, dim0: int, dim1: int) -> Tensor:
    return _swap_dims(x, dim0, dim1)


def moveaxis(
    x: Tensor,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Tensor:
    _src = list(source) if not isinstance(source, int) else source
    _dst = list(destination) if not isinstance(destination, int) else destination
    return lucid.movedim(x, _src, _dst)  # type: ignore[arg-type]


def adjoint(x: Tensor) -> Tensor:
    """Conjugate transpose of the last two dims.  Real-only for now."""
    if x.ndim < 2:
        raise ValueError("adjoint requires at least 2 dimensions")
    return _swap_dims(x, x.ndim - 2, x.ndim - 1)


def t(x: Tensor) -> Tensor:
    """Transpose for ≤2-D tensors (matches the reference framework's ``Tensor.t``)."""
    if x.ndim < 2:
        return x
    if x.ndim != 2:
        raise RuntimeError("t() expects a tensor with <= 2 dimensions")
    return _swap_dims(x, 0, 1)


# ── Stacks ─────────────────────────────────────────────────────────────────


def column_stack(tensors: Sequence[Tensor]) -> Tensor:
    """Stack 1-D tensors as columns (other ranks pass through unchanged)."""
    fixed = [t_i.unsqueeze(1) if t_i.ndim == 1 else t_i for t_i in tensors]
    return lucid.cat(fixed, 1)


def row_stack(tensors: Sequence[Tensor]) -> Tensor:
    """Alias for ``vstack`` (reference-framework parity)."""
    return lucid.vstack(list(tensors))


def dstack(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate along the third axis, reshaping lower-rank inputs."""
    fixed: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.reshape(1, -1, 1)
        elif t_i.ndim == 2:
            t_i = t_i.unsqueeze(2)
        fixed.append(t_i)
    return lucid.cat(fixed, 2)


def atleast_1d(*tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
    out = [t_i.reshape(1) if t_i.ndim == 0 else t_i for t_i in tensors]
    return out[0] if len(out) == 1 else tuple(out)


def atleast_2d(*tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
    out: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.unsqueeze(0)
        out.append(t_i)
    return out[0] if len(out) == 1 else tuple(out)


def atleast_3d(*tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
    out: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.reshape(1, -1, 1)
        elif t_i.ndim == 2:
            t_i = t_i.unsqueeze(2)
        out.append(t_i)
    return out[0] if len(out) == 1 else tuple(out)


# ── Splits ─────────────────────────────────────────────────────────────────


def _split_along(
    x: Tensor,
    indices_or_sections: int | Sequence[int],
    dim: int,
) -> list[Tensor]:
    """Convert NumPy-style splits to lucid's size-list form."""
    if isinstance(indices_or_sections, int):
        n = x.shape[dim]
        k = indices_or_sections
        base, extra = divmod(n, k)
        sizes = [base + 1] * extra + [base] * (k - extra)
        return lucid.split(x, sizes, dim)
    indices = list(indices_or_sections)
    split_sizes: list[int] = []
    prev = 0
    for idx in indices:
        split_sizes.append(idx - prev)
        prev = idx
    split_sizes.append(x.shape[dim] - prev)
    split_sizes = [s for s in split_sizes if s >= 0]
    return lucid.split(x, split_sizes, dim)


def vsplit(x: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    if x.ndim < 1:
        raise ValueError("vsplit requires at least 1-D input")
    return _split_along(x, indices_or_sections, 0)


def hsplit(x: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    return _split_along(x, indices_or_sections, 0 if x.ndim == 1 else 1)


def dsplit(x: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    if x.ndim < 3:
        raise ValueError("dsplit requires at least 3-D input")
    return _split_along(x, indices_or_sections, 2)


def tensor_split(
    x: Tensor,
    indices_or_sections: int | Sequence[int],
    dim: int = 0,
) -> list[Tensor]:
    return _split_along(x, indices_or_sections, dim)


# ── Misc ───────────────────────────────────────────────────────────────────


def take_along_dim(x: Tensor, indices: Tensor, dim: int) -> Tensor:
    """Gather elements at ``indices`` along ``dim``.

    lucid's ``gather`` signature is ``(input, indices, axis)``.
    """
    return lucid.gather(x, indices, dim)


def tril_indices(
    row: int,
    col: int | None = None,
    offset: int = 0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Indices of the lower-triangular part of an ``(row, col)`` matrix.

    Returns a 2-row tensor where row 0 holds row indices and row 1 holds
    column indices, in row-major order.  ``offset`` shifts the diagonal
    (positive = above, negative = below the main).
    """
    if col is None:
        col = row
    rows: list[int] = []
    cols: list[int] = []
    for i in range(row):
        for j in range(col):
            if j - i <= offset:
                rows.append(i)
                cols.append(j)
    out_dtype: DTypeLike = dtype if dtype is not None else lucid.int64
    return lucid.tensor([rows, cols], dtype=out_dtype, device=device)


def triu_indices(
    row: int,
    col: int | None = None,
    offset: int = 0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Indices of the upper-triangular part of an ``(row, col)`` matrix.

    Mirrors :func:`tril_indices` with the inequality flipped: keeps
    entries where ``j - i >= offset``.
    """
    if col is None:
        col = row
    rows: list[int] = []
    cols: list[int] = []
    for i in range(row):
        for j in range(col):
            if j - i >= offset:
                rows.append(i)
                cols.append(j)
    out_dtype: DTypeLike = dtype if dtype is not None else lucid.int64
    return lucid.tensor([rows, cols], dtype=out_dtype, device=device)


def combinations(
    input: Tensor,
    r: int = 2,
    with_replacement: bool = False,
) -> Tensor:
    """All ``r``-length combinations of the elements of a 1-D ``input``.

    Returns shape ``(C, r)`` where ``C = C(n, r)`` (or ``C(n+r-1, r)``
    when ``with_replacement=True``).  Output dtype follows ``input``.
    """
    import itertools as _it

    if input.ndim != 1:
        raise ValueError("combinations: input must be 1-D")
    n = int(input.shape[0])
    py_vals = [input[i].item() for i in range(n)]
    iterator = (
        _it.combinations_with_replacement(py_vals, r)
        if with_replacement
        else _it.combinations(py_vals, r)
    )
    rows = [list(combo) for combo in iterator]
    if not rows:
        return lucid.zeros(0, r, dtype=input.dtype, device=input.device)
    return lucid.tensor(rows, dtype=input.dtype, device=input.device)


def rot90(x: Tensor, k: int = 1, dims: Sequence[int] = (0, 1)) -> Tensor:
    """Rotate by 90° in the plane defined by ``dims``, ``k`` times."""
    d0, d1 = dims[0], dims[1]
    k = k % 4
    if k == 0:
        return x
    if k == 1:
        return _swap_dims(lucid.flip(x, [d1]), d0, d1)  # type: ignore[list-item]
    if k == 2:
        return lucid.flip(x, list(dims))  # type: ignore[arg-type]
    return _swap_dims(lucid.flip(x, [d0]), d0, d1)  # type: ignore[list-item]


__all__ = [
    "swapaxes",
    "swapdims",
    "moveaxis",
    "adjoint",
    "t",
    "column_stack",
    "row_stack",
    "dstack",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "vsplit",
    "hsplit",
    "dsplit",
    "tensor_split",
    "take_along_dim",
    "tril_indices",
    "triu_indices",
    "combinations",
    "rot90",
]
