"""NaN-safe reductions composed from ``where`` + standard reductions."""

import math
from typing import Sequence, TYPE_CHECKING

import lucid

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def nansum(
    x: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Sum of ``x`` treating NaN as zero."""
    safe = lucid.where(lucid.isnan(x), lucid.full_like(x, 0.0), x)
    if dim is None:
        return lucid.sum(safe)
    return lucid.sum(safe, dim, keepdim)


def nanmean(
    x: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Mean of ``x`` ignoring NaN entries (denominator is the non-NaN count)."""
    mask = lucid.isnan(x)
    safe = lucid.where(mask, lucid.full_like(x, 0.0), x)
    # Count non-NaN via where: 1.0 where not NaN, 0.0 where NaN
    not_nan = lucid.where(mask, lucid.full_like(x, 0.0), lucid.full_like(x, 1.0))
    if dim is None:
        return lucid.sum(safe) / lucid.sum(not_nan)
    return lucid.sum(safe, dim, keepdim) / lucid.sum(not_nan, dim, keepdim)


def nanmedian(
    x: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Median ignoring NaN entries.

    NaN entries are replaced with +inf before sorting, so they sink to
    the high end.  The median index is ``(non_nan_count - 1) // 2``
    (lower-median convention, matching the reference framework for odd counts).
    """
    big = lucid.full_like(x, math.inf)
    safe = lucid.where(lucid.isnan(x), big, x)
    nan_mask = lucid.isnan(x)
    not_nan = lucid.where(nan_mask, lucid.full_like(x, 0.0), lucid.full_like(x, 1.0))

    if dim is None:
        flat = safe.reshape(-1)
        sorted_vals = lucid.sort(flat)
        nn = int(not_nan.sum().item())
        if nn == 0:
            return lucid.tensor(math.nan, dtype=x.dtype, device=x.device)
        return sorted_vals[(nn - 1) // 2]

    sorted_vals = lucid.sort(safe, dim)
    nn = int(not_nan.sum().item() / max(1, x.numel() / x.shape[dim]))
    idx = max(0, (nn - 1) // 2)
    out = sorted_vals.index_select(dim, lucid.tensor([idx], dtype=lucid.int64))
    return out if keepdim else out.squeeze(dim)


def count_nonzero(
    x: Tensor,
    dim: int | Sequence[int] | None = None,
) -> Tensor:
    """Count non-zero elements along ``dim`` (or whole tensor when ``None``).

    Output dtype is int64 — the count semantics match the reference
    framework's ``count_nonzero``.

    The current CpuBackend lacks ``where(bool, i64, i64)`` and
    ``astype(bool → i64)``; we work in F32 (a 1.0/0.0 mask), reduce, then
    cast the resulting scalar / tensor to int64 via ``astype(F32 → I64)``,
    which the backend does support.
    """
    one_f = lucid.ones_like(x, dtype=lucid.float32)
    zero_f = lucid.zeros_like(x, dtype=lucid.float32)
    mask = lucid.where(x != lucid.zeros_like(x), one_f, zero_f)
    counts = lucid.sum(mask) if dim is None else lucid.sum(mask, dim, False)
    return counts.to(dtype=lucid.int64)


__all__ = ["nansum", "nanmean", "nanmedian", "count_nonzero"]
