"""NaN-safe reductions composed from ``where`` + standard reductions."""

from typing import Sequence, TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._ops._dims import _wrap_dim
from lucid._ops._reduce import _reduction_axes, _run_with_dtype
from lucid._types import DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def nansum(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike | None = None,
) -> Tensor:
    r"""Sum the tensor, treating NaN entries as zero.

    A NaN-safe variant of :func:`lucid.sum`. NaN values are replaced by
    ``0`` before the reduction, so they neither contaminate the result
    nor contribute to it.

    Parameters
    ----------
    input : Tensor
        Input tensor (any floating-point dtype).
    dim : int | Sequence[int] | None, optional
        Axis or axes along which to sum. ``None`` (default) reduces over
        the entire tensor.
    keepdim : bool, optional
        If ``True``, retains the reduced dimensions with size 1.
        Defaults to ``False``.
    dtype : DTypeLike | None, optional
        Target dtype for the reduction. When given, the input is cast to
        ``dtype`` before summing.

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
    tensor(4.)
    """
    axes = _reduction_axes(input._impl, dim, "nansum")
    safe = _C_engine.where(
        _C_engine.isnan(input._impl), _C_engine.zeros_like(input._impl), input._impl
    )
    return _wrap(
        _run_with_dtype(
            safe,
            dtype,
            lambda xi: _C_engine.sum(xi, axes, bool(keepdim)),
            op="nansum",
        )
    )


def _nanmean_impl(
    x_impl: _C_engine.TensorImpl, axes: list[int], keepdim: bool
) -> _C_engine.TensorImpl:
    mask = _C_engine.isnan(x_impl)
    zero = _C_engine.zeros_like(x_impl)
    one = _C_engine.ones_like(x_impl)
    safe = _C_engine.where(mask, zero, x_impl)
    not_nan = _C_engine.where(mask, zero, one)
    return _C_engine.div(
        _C_engine.sum(safe, axes, keepdim),
        _C_engine.sum(not_nan, axes, keepdim),
    )


def nanmean(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike | None = None,
) -> Tensor:
    r"""Mean of the tensor, ignoring NaN entries.

    A NaN-safe variant of :func:`lucid.mean`. Both the numerator (sum)
    and the denominator (element count) are computed only over the
    non-NaN entries.

    Parameters
    ----------
    input : Tensor
        Input tensor (any floating-point dtype).
    dim : int | Sequence[int] | None, optional
        Axis or axes along which to take the mean. ``None`` (default)
        reduces over the entire tensor.
    keepdim : bool, optional
        If ``True``, retains the reduced dimensions with size 1.
        Defaults to ``False``.
    dtype : DTypeLike | None, optional
        Target floating-point or complex dtype for the reduction. When
        given, ``input`` is cast to ``dtype`` before averaging.

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
    tensor(2.)
    """
    axes = _reduction_axes(input._impl, dim, "nanmean")
    return _wrap(
        _run_with_dtype(
            input._impl,
            dtype,
            lambda xi: _nanmean_impl(xi, axes, bool(keepdim)),
            op="nanmean",
            inexact_only=True,
        )
    )


def nanmedian(
    input: Tensor,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Median of a tensor, ignoring NaN entries.

    Computes the median over the requested axis (or the entire tensor
    when ``dim`` is ``None``), treating NaN entries as missing data
    rather than propagating them.  This is the NaN-safe counterpart of
    a regular ``median`` reduction.

    Parameters
    ----------
    input : Tensor
        Input tensor of a floating or integer dtype.
    dim : int | None, optional
        Reduction axis.  ``None`` (default) flattens ``input`` first.
    keepdim : bool, optional
        If ``True``, retain the reduced axis as a singleton in the
        output shape.  Ignored when ``dim is None``.  Defaults to
        ``False``.

    Returns
    -------
    Tensor or tuple of Tensor
        With ``dim is None``, the scalar median of the non-NaN entries.
        With ``dim`` given, ``(values, indices)``: the median of every
        slice along ``dim`` and the ``int64`` position of that median in
        ``input``, both the input shape with ``dim`` removed (or kept as
        size 1 if ``keepdim=True``).

    Raises
    ------
    NotImplementedError
        For a ``bool`` input.
    IndexError
        When ``dim`` is out of range, or the reduced axis has length 0.

    Notes
    -----
    Every slice keeps its own count :math:`n` of non-NaN entries.  NaN
    sorts last, so the median is the entry at sorted position
    :math:`\lfloor (n - 1) / 2 \rfloor` — the lower median for an even
    count, matching the reference framework.  A slice that is all NaN
    has no number to take and answers NaN, at the position of its first
    entry.  Integer inputs hold no NaN and take the plain lower median.

    The gradient of a ``dim`` reduction flows to the entry ``indices``
    names.  The whole-tensor median shares its gradient evenly between
    every entry equal to it (every NaN, when the answer is NaN), as the
    reference does.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([1.0, math.nan, 3.0, 5.0])
    >>> lucid.nanmedian(x)
    tensor(3.)
    >>> values, indices = lucid.nanmedian(
    ...     lucid.tensor([[1.0, math.nan, math.nan], [1.0, 2.0, 3.0]]), 1
    ... )
    >>> values
    tensor([1., 2.])
    >>> indices
    tensor([0, 1], dtype=lucid.int64)
    """
    if input.dtype == lucid.bool_:
        # As the reference does: a median of booleans is not defined here,
        # and the sort underneath accepts bool, so it has to be said.
        raise NotImplementedError(
            "nanmedian does not accept bool — cast to an integer dtype"
        )
    if dim is None:
        return _nanmedian_all(input)
    return _nanmedian_along(input, dim, keepdim)


def _nanmedian_along(x: Tensor, dim: int, keepdim: bool) -> tuple[Tensor, Tensor]:
    """``(values, indices)`` of the NaN-skipping median of every slice.

    The position is read per slice from that slice's own non-NaN count —
    a tensor, so nothing leaves the device and no slice borrows another's
    count.
    """
    d = _wrap_dim(dim, x.ndim, "nanmedian")
    if x.ndim == 0:
        return _nanmedian_along(x.reshape(1), 0, False)
    n = x.shape[d]
    if n == 0:
        raise IndexError(
            f"nanmedian: the reduced dim {dim} has length 0 — there is no "
            "median of nothing"
        )
    # The engine's sort puts NaN after every number, ``+inf`` included,
    # and keeps tied entries in their input order.
    order = lucid.argsort(x, d)
    if lucid.is_floating_point(x):
        count = lucid.sum(~lucid.isnan(x), d, True)
        pos = lucid.clamp((count - 1) // 2, min=0)
    else:
        pos = lucid.full_like(order.narrow(d, 0, 1), (n - 1) // 2)
    indices = lucid.take_along_dim(order, pos, d)
    values = lucid.take_along_dim(x, indices, d)
    if not keepdim:
        return values.squeeze(d), indices.squeeze(d)
    return values, indices


def _nanmedian_all(x: Tensor) -> Tensor:
    """The NaN-skipping median of every entry of ``x``."""
    flat = x.reshape(-1)
    if flat.shape[0] == 0:
        # There is no entry, so the answer is NaN.  A fresh constant would
        # cut the autograd edge; the mean of no entries is NaN as well.
        return lucid.mean(flat)
    value, _ = _nanmedian_along(flat, 0, False)
    if not x.requires_grad:
        return value
    # The reference hands every entry equal to the median an even share of
    # the gradient, not only the one the sort happened to put in the middle.
    picked = value.detach()
    tie = (flat == picked) | (lucid.isnan(flat) & lucid.isnan(picked))
    share = (1.0 / lucid.sum(tie).to(lucid.float32)).to(flat.dtype)
    ties = lucid.where(tie, flat, lucid.zeros_like(flat))
    # ``ties - ties.detach()`` is zero with a unit derivative, so ``picked``
    # keeps its exact value — except at ``inf`` or NaN, where it is NaN.
    # Those answers are spread instead: a sum of copies of ``inf`` or NaN,
    # scaled, is still exactly that.
    exact = picked + lucid.sum(ties - ties.detach()) * share
    spread = lucid.sum(ties) * share
    return lucid.where(lucid.isfinite(picked), exact, spread)


def count_nonzero(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
) -> Tensor:
    """Count non-zero elements along ``dim`` (or whole tensor when ``None``).

    Parameters
    ----------
    input : Tensor
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
    The sum of the bool mask ``x != 0``, which is int64 and exact.  It
    used to build a float32 ``1.0/0.0`` mask with ``where`` and cast the
    float sum back — three full-size temporaries and ten times a plain
    reduction, from a time the CPU could not sum a bool.
    """
    mask = input != 0
    _dim2 = None if dim is None else _reduction_axes(input._impl, dim, "count_nonzero")
    return lucid.sum(mask) if _dim2 is None else lucid.sum(mask, _dim2, False)


def amax(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Reduce ``input`` to per-slice maxima along ``dim`` — values only.

    Multi-axis sibling of :func:`lucid.max` that returns *only* the
    reduced values (never the indices) and accepts a tuple of axes
    for a single fused reduction.  Use when you don't need the
    ``(values, indices)`` tuple :func:`lucid.max` returns.

    Parameters
    ----------
    input : Tensor
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
        return lucid.max(input)
    _dim = _reduction_axes(input._impl, dim, "amax")
    return lucid.max(input, _dim, keepdim)


def amin(
    input: Tensor,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Reduce ``input`` to per-slice minima along ``dim`` — values only.

    Mirror image of :func:`amax`.  Accepts a tuple of axes for a fused
    multi-axis reduction; returns just the minimum values (never the
    indices).

    Parameters
    ----------
    input : Tensor
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
        return lucid.min(input)
    _dim = _reduction_axes(input._impl, dim, "amin")
    return lucid.min(input, _dim, keepdim)


__all__ = ["nansum", "nanmean", "nanmedian", "count_nonzero", "amax", "amin"]

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call. Imported at the bottom because this
# module is inside the cycle ``Tensor`` closes.
from lucid._tensor.tensor import Tensor  # noqa: E402,F811
