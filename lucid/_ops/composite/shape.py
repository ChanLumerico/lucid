"""Shape-manipulation composites: axis swaps, stacks, splits, and the
miscellaneous fillers (``rot90``, ``vander``, ``take_along_dim``).
"""

from typing import Sequence, TYPE_CHECKING

import lucid
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
    return lucid.movedim(x, source, destination)


def adjoint(x: Tensor) -> Tensor:
    """Conjugate transpose of the last two dims.  Real-only for now."""
    if x.ndim < 2:
        raise ValueError("adjoint requires at least 2 dimensions")
    return _swap_dims(x, x.ndim - 2, x.ndim - 1)


def t(x: Tensor) -> Tensor:
    """Transpose for ≤2-D tensors (PyTorch's ``Tensor.t`` semantics)."""
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
    """Alias for ``vstack`` (PyTorch parity)."""
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
    sizes: list[int] = []
    prev = 0
    for idx in indices:
        sizes.append(idx - prev)
        prev = idx
    sizes.append(x.shape[dim] - prev)
    sizes = [s for s in sizes if s >= 0]
    return lucid.split(x, sizes, dim)


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


def rot90(x: Tensor, k: int = 1, dims: Sequence[int] = (0, 1)) -> Tensor:
    """Rotate by 90° in the plane defined by ``dims``, ``k`` times."""
    d0, d1 = dims[0], dims[1]
    k = k % 4
    if k == 0:
        return x
    if k == 1:
        return _swap_dims(lucid.flip(x, [d1]), d0, d1)
    if k == 2:
        return lucid.flip(x, list(dims))
    return _swap_dims(lucid.flip(x, [d0]), d0, d1)


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
    "rot90",
]
