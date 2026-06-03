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
    r"""Sum the tensor, treating NaN entries as zero.

    A NaN-safe variant of :func:`lucid.sum`. NaN values are replaced by
    ``0`` before the reduction, so they neither contaminate the result
    nor contribute to it.

    Parameters
    ----------
    x : Tensor
        Input tensor (any floating-point dtype).
    dim : int | Sequence[int] | None, optional
        Axis or axes along which to sum. ``None`` (default) reduces over
        the entire tensor.
    keepdim : bool, optional
        If ``True``, retains the reduced dimensions with size 1.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        Reduced tensor.

    Notes
    -----
    With :math:`\mathcal{S}` the set of indices being reduced and
    :math:`\mathbb{1}_{x_i \text{ is NaN}}` an indicator,

    .. math::

        \text{nansum}(x) =
        \sum_{i \in \mathcal{S}}
        (1 - \mathbb{1}_{x_i \text{ is NaN}}) \cdot x_i.

    The gradient at NaN positions is zero — those entries contribute
    nothing to the forward sum, so they cannot influence it through
    perturbation.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([1.0, math.nan, 3.0])
    >>> lucid.nansum(x)
    Tensor(4.)
    """
    safe = lucid.where(lucid.isnan(x), lucid.full_like(x, 0.0), x)
    if dim is None:
        return lucid.sum(safe)
    _dim = list(dim) if not isinstance(dim, int) else dim
    return lucid.sum(safe, _dim, keepdim)


def nanmean(
    x: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Mean of the tensor, ignoring NaN entries.

    A NaN-safe variant of :func:`lucid.mean`. Both the numerator (sum)
    and the denominator (element count) are computed only over the
    non-NaN entries.

    Parameters
    ----------
    x : Tensor
        Input tensor (any floating-point dtype).
    dim : int | Sequence[int] | None, optional
        Axis or axes along which to take the mean. ``None`` (default)
        reduces over the entire tensor.
    keepdim : bool, optional
        If ``True``, retains the reduced dimensions with size 1.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        Reduced tensor.

    Notes
    -----
    With :math:`\mathcal{S}` the set of indices being reduced and
    :math:`\mathbb{1}_i = 1 - \mathbb{1}_{x_i \text{ is NaN}}`,

    .. math::

        \text{nanmean}(x) =
        \frac{\sum_{i \in \mathcal{S}} \mathbb{1}_i \cdot x_i}
             {\sum_{i \in \mathcal{S}} \mathbb{1}_i}.

    A slice that is entirely NaN produces ``0 / 0 = NaN``; this is the
    standard convention. The gradient at NaN positions is zero.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([1.0, math.nan, 3.0])
    >>> lucid.nanmean(x)
    Tensor(2.)
    """
    mask = lucid.isnan(x)
    safe = lucid.where(mask, lucid.full_like(x, 0.0), x)
    # Count non-NaN via where: 1.0 where not NaN, 0.0 where NaN
    not_nan = lucid.where(mask, lucid.full_like(x, 0.0), lucid.full_like(x, 1.0))
    if dim is None:
        return lucid.sum(safe) / lucid.sum(not_nan)
    _dim = list(dim) if not isinstance(dim, int) else dim
    return lucid.sum(safe, _dim, keepdim) / lucid.sum(not_nan, _dim, keepdim)


def nanmedian(
    x: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor:
    r"""Median of a tensor, ignoring NaN entries.

    Computes the median over the requested axis (or the entire tensor
    when ``dim`` is ``None``), treating NaN entries as missing data
    rather than propagating them.  This is the NaN-safe counterpart of
    a regular ``median`` reduction.

    Parameters
    ----------
    x : Tensor
        Input tensor of any floating dtype.
    dim : int | None, optional
        Reduction axis.  ``None`` (default) flattens ``x`` first.
    keepdim : bool, optional
        If ``True``, retain the reduced axis as a singleton in the
        output shape.  Ignored when ``dim is None``.  Defaults to
        ``False``.

    Returns
    -------
    Tensor
        Median value(s) over the non-NaN entries.  Scalar when
        ``dim is None``; otherwise the input shape with the reduction
        axis removed (or kept as size 1 if ``keepdim=True``).

    Notes
    -----
    Algorithmic outline:

    1. Replace NaN entries with :math:`+\infty` so they sort to the
       high end and never become the chosen median.
    2. Sort along the reduction axis.
    3. Pick element at index
       :math:`\lfloor (n_{\text{not-nan}} - 1) / 2 \rfloor` — the
       lower-median convention for even counts, matching the
       reference framework's default behaviour.

    If *all* entries along the reduction are NaN the result is NaN
    (no valid sample to take).

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([1.0, math.nan, 3.0, 5.0])
    >>> lucid.nanmedian(x)
    Tensor(3.)
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

    Parameters
    ----------
    x : Tensor
        Input.  Non-zero is determined by element-wise comparison to
        zero (NaN counts as non-zero).
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) reduces the entire
        tensor to a 0-D scalar.

    Returns
    -------
    Tensor
        ``int64`` count(s).  Shape mirrors :func:`lucid.sum` over the
        same ``dim`` argument.

    Notes
    -----
    Composite implementation: the CPU backend lacks
    ``where(bool, i64, i64)`` and ``astype(bool → i64)``, so the op
    builds a float32 ``1.0/0.0`` mask, reduces, and ``astype``-casts
    the float scalar back to int64 (a path the backend does support).
    Slightly more allocations than a native int reduction but
    semantically identical.
    """
    one_f = lucid.ones_like(x, dtype=lucid.float32)
    zero_f = lucid.zeros_like(x, dtype=lucid.float32)
    mask = lucid.where(x != lucid.zeros_like(x), one_f, zero_f)
    _dim2 = list(dim) if dim is not None and not isinstance(dim, int) else dim
    counts = lucid.sum(mask) if _dim2 is None else lucid.sum(mask, _dim2, False)
    return counts.to(dtype=lucid.int64)


def amax(
    x: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Reduce ``x`` to per-slice maxima along ``dim`` — values only.

    Multi-axis sibling of :func:`lucid.max` that returns *only* the
    reduced values (never the indices) and accepts a tuple of axes
    for a single fused reduction.  Use when you don't need the
    ``(values, indices)`` tuple :func:`lucid.max` returns.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) reduces over every
        axis and returns a 0-D scalar tensor.
    keepdim : bool, optional
        Retain the reduced axes as size-1 dimensions.  Default
        ``False``.

    Returns
    -------
    Tensor
        Maxima along ``dim``; shape depends on ``dim`` and ``keepdim``.

    See Also
    --------
    amin : analogous minimum reduction.
    lucid.max : returns ``(values, indices)`` for a single axis.
    """
    if dim is None:
        return lucid.max(x)
    _dim = list(dim) if not isinstance(dim, int) else dim
    return lucid.max(x, _dim, keepdim)


def amin(
    x: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Reduce ``x`` to per-slice minima along ``dim`` — values only.

    Mirror image of :func:`amax`.  Accepts a tuple of axes for a fused
    multi-axis reduction; returns just the minimum values (never the
    indices).

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) reduces over every
        axis to a 0-D scalar.
    keepdim : bool, optional
        Retain the reduced axes as size-1 dimensions.  Default
        ``False``.

    Returns
    -------
    Tensor
        Minima along ``dim``.

    See Also
    --------
    amax : analogous maximum reduction.
    lucid.min : returns ``(values, indices)`` for a single axis.
    """
    if dim is None:
        return lucid.min(x)
    _dim = list(dim) if not isinstance(dim, int) else dim
    return lucid.min(x, _dim, keepdim)


__all__ = ["nansum", "nanmean", "nanmedian", "count_nonzero", "amax", "amin"]
