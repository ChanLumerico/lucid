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
    (lower-median convention, matching PyTorch for odd counts).
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


__all__ = ["nansum", "nanmean", "nanmedian"]
