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
    r"""Return ``x`` with axes ``axis0`` and ``axis1`` exchanged.

    NumPy-style spelling of pairwise axis transposition.  Verbose alias of
    :func:`swapdims`; both names refer to the same composite.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    axis0 : int
        First axis to swap.  Negative values count from the end.
    axis1 : int
        Second axis to swap.  Negative values count from the end.

    Returns
    -------
    Tensor
        Tensor with the same data as ``x`` but with ``axis0`` and
        ``axis1`` exchanged.  Where possible Lucid returns a view; if
        the underlying storage is non-contiguous, a freshly materialised
        tensor is returned (still satisfying value semantics — Lucid
        never aliases in a way that would surprise in-place writers).

    Notes
    -----
    For a tensor with shape :math:`(s_0, \dots, s_{n-1})`, the output
    shape is

    .. math::

        s'_i =
        \begin{cases}
            s_{a_1}, & i = a_0, \\
            s_{a_0}, & i = a_1, \\
            s_i,     & \text{otherwise.}
        \end{cases}

    Some reference frameworks document ``swapaxes`` as having in-place
    view semantics — in Lucid it is always a (possibly fresh) value-view.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros((2, 3, 4))
    >>> lucid.swapaxes(x, 0, 2).shape
    (4, 3, 2)
    """
    return _swap_dims(x, axis0, axis1)


def swapdims(x: Tensor, dim0: int, dim1: int) -> Tensor:
    r"""Swap two named dimensions of a tensor.

    Returns a view (or, if not contiguous, a freshly materialised tensor)
    whose dimensions ``dim0`` and ``dim1`` have been exchanged. All other
    dimensions retain their position.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim0 : int
        First dimension to swap. Negative values count from the end.
    dim1 : int
        Second dimension to swap. Negative values count from the end.

    Returns
    -------
    Tensor
        Tensor with the same data as ``x`` but with axes ``dim0`` and
        ``dim1`` exchanged.

    Notes
    -----
    For a tensor with shape :math:`(s_0, \dots, s_{n-1})`, the result
    has shape

    .. math::

        s'_i =
        \begin{cases}
            s_{d_1}, & i = d_0, \\
            s_{d_0}, & i = d_1, \\
            s_i,     & \text{otherwise.}
        \end{cases}

    Equivalent to :func:`swapaxes`; both names are provided for parity
    with NumPy and reference-framework conventions.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros((2, 3, 4))
    >>> lucid.swapdims(x, 0, 2).shape
    (4, 3, 2)
    """
    return _swap_dims(x, dim0, dim1)


def moveaxis(
    x: Tensor,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Tensor:
    r"""Move one or more axes to new positions.

    More flexible than :func:`swapaxes` — instead of pairwise swapping,
    ``moveaxis`` re-positions a whole set of axes while preserving the
    relative order of the remaining axes.  Thin wrapper around
    :func:`lucid.movedim` that accepts either a single ``int`` or a
    sequence on both endpoints.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    source : int | Sequence[int]
        Original position(s) of the axes to move.  Negative values count
        from the end.
    destination : int | Sequence[int]
        Final position(s) for each moved axis.  Must have the same length
        as ``source`` (after normalisation) and be a permutation-compatible
        target set.

    Returns
    -------
    Tensor
        Tensor whose axes have been re-ordered as requested.

    Notes
    -----
    Equivalent to applying a permutation :math:`\pi` to the axes such
    that ``axis at source[i]`` ends up at ``destination[i]`` while the
    other axes slide to fill in the gaps preserving their relative
    order.  Useful for vectorised code that needs to ferry a batch
    dimension across the rank of a tensor.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros((2, 3, 4, 5))
    >>> lucid.moveaxis(x, 0, -1).shape
    (3, 4, 5, 2)
    >>> lucid.moveaxis(x, [0, 1], [-1, -2]).shape
    (4, 5, 3, 2)
    """
    _src = list(source) if not isinstance(source, int) else source
    _dst = list(destination) if not isinstance(destination, int) else destination
    return lucid.movedim(x, _src, _dst)  # type: ignore[arg-type]


def adjoint(x: Tensor) -> Tensor:
    r"""Conjugate (Hermitian) transpose of the trailing two dimensions.

    For real-valued tensors this is identical to a plain transpose of the
    last two axes. For complex tensors (when supported), the entries are
    additionally conjugated, yielding the Hermitian transpose
    :math:`A^{*} = \overline{A^{T}}`.

    Parameters
    ----------
    x : Tensor
        Input tensor with at least 2 dimensions.

    Returns
    -------
    Tensor
        Tensor with the last two axes swapped (and complex conjugated
        when applicable).

    Raises
    ------
    ValueError
        If ``x.ndim < 2``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \mathbf{A}^{*} = \overline{\mathbf{A}^{T}}.

    For real inputs the conjugation step is a no-op, so this is simply
    a permutation of the last two axes. Batch dimensions are preserved.

    Examples
    --------
    >>> import lucid
    >>> A = lucid.tensor([[1., 2., 3.], [4., 5., 6.]])
    >>> lucid.adjoint(A)
    Tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    """
    if x.ndim < 2:
        raise ValueError("adjoint requires at least 2 dimensions")
    return _swap_dims(x, x.ndim - 2, x.ndim - 1)


def t(x: Tensor) -> Tensor:
    r"""Short-form transpose for tensors of rank at most 2.

    A convenience analogue of MATLAB's ``A'`` or the reference
    framework's ``Tensor.t()``: returns the transpose of a 2-D matrix,
    or the input unchanged for 0-D and 1-D tensors.

    Parameters
    ----------
    x : Tensor
        Input tensor with at most 2 dimensions.

    Returns
    -------
    Tensor
        For ``ndim <= 1``, returns ``x`` unchanged.
        For ``ndim == 2``, returns the 2-D transpose.

    Raises
    ------
    RuntimeError
        If ``x`` has 3 or more dimensions.

    Notes
    -----
    Defined as

    .. math::

        \text{t}(x) =
        \begin{cases}
            x,            & \text{ndim}(x) \leq 1, \\
            x^{T},        & \text{ndim}(x) = 2.
        \end{cases}

    Use :func:`adjoint` or :func:`lucid.transpose` for batched / higher
    rank inputs.

    Examples
    --------
    >>> import lucid
    >>> A = lucid.tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> lucid.t(A).shape
    (2, 3)
    """
    if x.ndim < 2:
        return x
    if x.ndim != 2:
        raise RuntimeError("t() expects a tensor with <= 2 dimensions")
    return _swap_dims(x, 0, 1)


# ── Stacks ─────────────────────────────────────────────────────────────────


def column_stack(tensors: Sequence[Tensor]) -> Tensor:
    r"""Stack tensors as columns of a 2-D matrix.

    Each 1-D tensor is first promoted to a column vector of shape
    ``(N, 1)``; tensors with rank ``>= 2`` are passed through unchanged.
    The results are then concatenated along axis 1.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Tensors to stack. 1-D tensors are reshaped to column form;
        higher-rank tensors must already have matching first-axis size.

    Returns
    -------
    Tensor
        2-D tensor formed by concatenating the (possibly promoted)
        inputs along axis 1.

    Notes
    -----
    For all-1-D inputs :math:`v^{(1)}, \dots, v^{(k)}` of length ``N``,
    the result has shape ``(N, k)``:

    .. math::

        \text{out}_{ij} = v^{(j)}_i.

    For higher-rank inputs, the first dimension is the "row count" along
    which axis-1 concatenation happens, matching NumPy semantics.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1., 2., 3.])
    >>> b = lucid.tensor([4., 5., 6.])
    >>> lucid.column_stack([a, b])
    Tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    """
    fixed = [t_i.unsqueeze(1) if t_i.ndim == 1 else t_i for t_i in tensors]
    return lucid.cat(fixed, 1)


def row_stack(tensors: Sequence[Tensor]) -> Tensor:
    r"""Stack tensors as rows — alias of :func:`lucid.vstack`.

    Verbose name provided for API parity. Each 1-D input is treated as a
    row and concatenated along axis 0.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Tensors to stack. 1-D tensors are promoted to ``(1, N)`` row
        vectors; higher-rank tensors must have matching column count.

    Returns
    -------
    Tensor
        Tensor formed by concatenating the (possibly promoted) inputs
        along axis 0.

    Notes
    -----
    For all-1-D inputs of length ``N``, the result has shape ``(k, N)``:

    .. math::

        \text{out}_{ij} = v^{(i)}_j.

    The 1-D-to-2-D promotion is the inverse of :func:`column_stack`'s
    promotion.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1., 2., 3.])
    >>> b = lucid.tensor([4., 5., 6.])
    >>> lucid.row_stack([a, b])
    Tensor([[1., 2., 3.],
            [4., 5., 6.]])
    """
    return lucid.vstack(list(tensors))


def dstack(tensors: Sequence[Tensor]) -> Tensor:
    r"""Stack tensors along the third (depth) axis.

    Reshapes lower-rank inputs to give them a depth dimension, then
    concatenates along axis 2. 0-D scalars become ``(1, 1, 1)``, 1-D
    vectors of length ``N`` become ``(1, N, 1)``, and 2-D matrices of
    shape ``(H, W)`` become ``(H, W, 1)``.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        Tensors to stack. 0-, 1-, and 2-D inputs are auto-promoted to
        3-D as described above; 3-D inputs pass through unchanged.

    Returns
    -------
    Tensor
        3-D tensor formed by concatenating the (possibly promoted)
        inputs along axis 2.

    Notes
    -----
    The promotion rules ensure the result is always at least 3-D, which
    is useful for stacking colour channels of images: ``dstack([R, G, B])``
    produces an ``(H, W, 3)`` tensor from three ``(H, W)`` planes.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([[1., 2.], [3., 4.]])
    >>> b = lucid.tensor([[5., 6.], [7., 8.]])
    >>> lucid.dstack([a, b]).shape
    (2, 2, 2)
    """
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
    r"""Promote each input to at least 1 dimension.

    Scalars (0-D tensors) are reshaped to length-1 vectors; tensors that
    already have rank :math:`\geq 1` pass through unchanged.  Commonly
    used as a defensive guard at the top of routines that vectorise over
    a leading axis.

    Parameters
    ----------
    *tensors : Tensor
        One or more input tensors of arbitrary rank.

    Returns
    -------
    Tensor | tuple[Tensor, ...]
        A single tensor when called with one argument, otherwise a tuple
        of the promoted tensors in the same order as the inputs.

    Notes
    -----
    Promotion rule per input ``t`` with original rank ``r``:

    .. math::

        \text{shape}'(t) =
        \begin{cases}
            (1,),               & r = 0, \\
            \text{shape}(t),    & r \geq 1.
        \end{cases}

    Mirrors NumPy's ``np.atleast_1d``.  Higher-rank ``atleast_2d`` and
    ``atleast_3d`` variants follow the same pattern with extra leading /
    trailing unit axes.

    Examples
    --------
    >>> import lucid
    >>> s = lucid.tensor(3.0)
    >>> v = lucid.tensor([1.0, 2.0])
    >>> a, b = lucid.atleast_1d(s, v)
    >>> a.shape, b.shape
    ((1,), (2,))
    """
    out = [t_i.reshape(1) if t_i.ndim == 0 else t_i for t_i in tensors]
    return out[0] if len(out) == 1 else tuple(out)


def atleast_2d(*tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
    r"""Promote each input to at least 2 dimensions.

    Scalars become shape ``(1, 1)``, 1-D tensors gain a leading unit axis
    to become ``(1, N)``; tensors that already have rank :math:`\geq 2`
    pass through unchanged.

    Parameters
    ----------
    *tensors : Tensor
        One or more input tensors of arbitrary rank.

    Returns
    -------
    Tensor | tuple[Tensor, ...]
        A single tensor when called with one argument, otherwise a tuple
        of the promoted tensors in the same order as the inputs.

    Notes
    -----
    Promotion rule per input ``t`` with original shape :math:`\mathbf{s}`:

    .. math::

        \text{shape}'(t) =
        \begin{cases}
            (1, 1),           & \text{ndim}(t) = 0, \\
            (1, N),           & \mathbf{s} = (N,), \\
            \mathbf{s},       & \text{ndim}(t) \geq 2.
        \end{cases}

    The leading axis insertion (rather than trailing) matches NumPy's
    ``np.atleast_2d`` convention.

    Examples
    --------
    >>> import lucid
    >>> v = lucid.tensor([1.0, 2.0, 3.0])
    >>> lucid.atleast_2d(v).shape
    (1, 3)
    """
    out: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.unsqueeze(0)
        out.append(t_i)
    return out[0] if len(out) == 1 else tuple(out)


def atleast_3d(*tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
    r"""Promote each input to at least 3 dimensions.

    Scalars become shape ``(1, 1, 1)``, 1-D tensors become ``(1, N, 1)``,
    2-D tensors gain a trailing unit axis to become ``(M, N, 1)``;
    tensors with rank :math:`\geq 3` pass through unchanged.

    Parameters
    ----------
    *tensors : Tensor
        One or more input tensors of arbitrary rank.

    Returns
    -------
    Tensor | tuple[Tensor, ...]
        A single tensor when called with one argument, otherwise a tuple
        of the promoted tensors in the same order as the inputs.

    Notes
    -----
    Promotion rule per input ``t``:

    .. math::

        \text{shape}'(t) =
        \begin{cases}
            (1, 1, 1),    & \text{ndim}(t) = 0, \\
            (1, N, 1),    & \text{shape}(t) = (N,), \\
            (M, N, 1),    & \text{shape}(t) = (M, N), \\
            \text{shape}(t), & \text{ndim}(t) \geq 3.
        \end{cases}

    Note the asymmetric padding: 1-D tensors get both a leading and a
    trailing unit axis (so they live in the middle, image-like axis),
    while 2-D tensors gain only a trailing channel axis.  This matches
    NumPy and is convenient for image-processing code that expects
    ``(H, W, C)`` arrays.

    Examples
    --------
    >>> import lucid
    >>> v = lucid.tensor([1.0, 2.0])
    >>> lucid.atleast_3d(v).shape
    (1, 2, 1)
    """
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
    r"""Split a tensor along its first (vertical) axis.

    NumPy-style vertical split: cuts the input into pieces along axis 0.
    With an integer ``k``, the tensor is divided into ``k`` near-equal
    pieces; with a sequence of indices, the cuts occur at those positions.

    Parameters
    ----------
    x : Tensor
        Input tensor with at least 1 dimension.
    indices_or_sections : int | Sequence[int]
        - ``int``: number of equal-sized splits. If the axis length is
          not divisible, the first ``axis_len % k`` pieces get one extra
          element.
        - ``Sequence[int]``: cut indices along axis 0.

    Returns
    -------
    list[Tensor]
        Sub-tensors whose stacking along axis 0 reproduces ``x``.

    Raises
    ------
    ValueError
        If ``x.ndim < 1``.

    Notes
    -----
    For a tensor of shape :math:`(N, \dots)` split into ``k`` near-equal
    pieces, each piece has shape :math:`(\lceil N/k \rceil, \dots)` or
    :math:`(\lfloor N/k \rfloor, \dots)`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(12).reshape(4, 3)
    >>> [s.shape for s in lucid.vsplit(x, 2)]
    [(2, 3), (2, 3)]
    """
    if x.ndim < 1:
        raise ValueError("vsplit requires at least 1-D input")
    return _split_along(x, indices_or_sections, 0)


def hsplit(x: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    r"""Split a tensor horizontally (along axis 1 for rank :math:`\geq 2`).

    NumPy-style horizontal split: cuts the input column-wise along axis 1
    for tensors of rank :math:`\geq 2`.  For 1-D inputs the axis collapses
    to 0, since there is only one dimension available.

    Parameters
    ----------
    x : Tensor
        Input tensor.  At least 1-D.
    indices_or_sections : int | Sequence[int]
        - ``int``: number of (near-)equal-sized splits.  If the axis
          length is not divisible by ``k``, the first ``axis_len % k``
          pieces get one extra element.
        - ``Sequence[int]``: cut indices along the split axis (axis 1
          for rank :math:`\geq 2`, axis 0 for 1-D).

    Returns
    -------
    list[Tensor]
        Sub-tensors whose concatenation along the split axis reproduces
        ``x``.

    Notes
    -----
    Companion to :func:`vsplit` (axis 0) and :func:`dsplit` (axis 2).
    The axis-selection rule

    .. math::

        \text{axis} =
        \begin{cases}
            0, & \text{ndim}(x) = 1, \\
            1, & \text{ndim}(x) \geq 2,
        \end{cases}

    matches NumPy's convention so that 1-D inputs behave intuitively
    rather than raising on the missing column axis.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(12).reshape(3, 4)
    >>> [s.shape for s in lucid.hsplit(x, 2)]
    [(3, 2), (3, 2)]
    >>> v = lucid.arange(6)
    >>> [s.shape for s in lucid.hsplit(v, 3)]
    [(2,), (2,), (2,)]
    """
    return _split_along(x, indices_or_sections, 0 if x.ndim == 1 else 1)


def dsplit(x: Tensor, indices_or_sections: int | Sequence[int]) -> list[Tensor]:
    r"""Split a tensor along its third (depth) axis.

    NumPy-style depth-wise split: cuts the input into pieces along axis 2.
    Mirrors :func:`vsplit` (axis 0) and :func:`hsplit` (axis 1).

    Parameters
    ----------
    x : Tensor
        Input tensor with at least 3 dimensions.
    indices_or_sections : int | Sequence[int]
        - ``int``: number of equal-sized splits along axis 2.
        - ``Sequence[int]``: cut indices along axis 2.

    Returns
    -------
    list[Tensor]
        Sub-tensors whose concatenation along axis 2 reproduces ``x``.

    Raises
    ------
    ValueError
        If ``x.ndim < 3``.

    Notes
    -----
    For a tensor of shape :math:`(\dots, D)` split into ``k`` near-equal
    pieces along the last (depth) axis, each piece has shape
    :math:`(\dots, \lceil D/k \rceil)` or :math:`(\dots, \lfloor D/k \rfloor)`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(24).reshape(2, 3, 4)
    >>> [s.shape for s in lucid.dsplit(x, 2)]
    [(2, 3, 2), (2, 3, 2)]
    """
    if x.ndim < 3:
        raise ValueError("dsplit requires at least 3-D input")
    return _split_along(x, indices_or_sections, 2)


def tensor_split(
    x: Tensor,
    indices_or_sections: int | Sequence[int],
    dim: int = 0,
) -> list[Tensor]:
    r"""Split a tensor along ``dim``, permitting unequal final-piece sizes.

    More permissive than :func:`lucid.split`: when ``indices_or_sections``
    is an integer ``k`` and ``x.shape[dim]`` is not divisible by ``k``,
    the first ``x.shape[dim] % k`` pieces receive one extra element each
    while the remaining pieces take the smaller floor size.  This mirrors
    NumPy's ``np.array_split`` semantics, where ``split`` raises on
    non-divisible counts but ``tensor_split`` quietly returns ragged
    pieces.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    indices_or_sections : int | Sequence[int]
        - ``int``: number of (near-)equal-sized splits.  The first
          ``axis_len % k`` pieces receive one extra element.
        - ``Sequence[int]``: cut indices along ``dim``.
    dim : int, optional
        Axis to split along.  Defaults to ``0``.

    Returns
    -------
    list[Tensor]
        Sub-tensors whose concatenation along ``dim`` reproduces ``x``.

    Notes
    -----
    For integer ``k`` with :math:`n = \text{shape}(x)[\text{dim}]` and
    :math:`b = \lfloor n / k \rfloor`, :math:`r = n \bmod k`, the piece
    sizes are

    .. math::

        \underbrace{b + 1, \dots, b + 1}_{r\text{ terms}},\;
        \underbrace{b, \dots, b}_{k - r\text{ terms}}.

    Contrast with :func:`lucid.split`, which raises ``ValueError`` when
    ``n`` is not a multiple of ``k``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.arange(10)
    >>> [s.shape for s in lucid.tensor_split(x, 3)]
    [(4,), (3,), (3,)]
    >>> [s.shape for s in lucid.tensor_split(x, [2, 5])]
    [(2,), (3,), (5,)]
    """
    return _split_along(x, indices_or_sections, dim)


# ── Misc ───────────────────────────────────────────────────────────────────


def take_along_dim(x: Tensor, indices: Tensor, dim: int) -> Tensor:
    r"""Gather elements from ``x`` at positions ``indices`` along ``dim``.

    Advanced indexing primitive analogous to ``np.take_along_axis``:
    selects one element from ``x`` for every entry in ``indices``,
    broadcasting the remaining (non-``dim``) axes between the two
    tensors.  Thin wrapper around :func:`lucid.gather` to align with
    the reference-framework spelling.

    Parameters
    ----------
    x : Tensor
        Source tensor.
    indices : Tensor
        Integer tensor of positions along ``dim``.  Its shape must be
        broadcast-compatible with ``x`` on every axis other than ``dim``;
        the size along ``dim`` controls the size of the output along
        that axis.
    dim : int
        Axis along which to gather.  Negative values count from the end.

    Returns
    -------
    Tensor
        Tensor with the broadcast shape of ``x`` and ``indices`` (with
        the size along ``dim`` taken from ``indices``).  Dtype matches
        ``x``.

    Notes
    -----
    For 1-D inputs the operation reduces to plain integer indexing.  For
    higher-rank inputs, the result satisfies

    .. math::

        \text{out}[i_0, \dots, i_{d-1}, j, i_{d+1}, \dots] =
        x[i_0, \dots, i_{d-1},\;
          \text{indices}[i_0, \dots, i_{d-1}, j, i_{d+1}, \dots],\;
          i_{d+1}, \dots].

    Typical uses include "gather the top-k elements per row" patterns
    after :func:`lucid.argsort` / :func:`lucid.topk`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([[1.0, 2.0, 3.0],
    ...                   [4.0, 5.0, 6.0]])
    >>> idx = lucid.tensor([[2, 0], [1, 2]])
    >>> lucid.take_along_dim(x, idx, dim=1)
    Tensor([[3., 1.],
            [5., 6.]])
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
    r"""Rotate a tensor by 90° in a chosen plane.

    Applies ``k`` successive 90° rotations in the plane spanned by the
    two axes ``dims = (d_0, d_1)``. The rotation direction is from
    ``d_0`` toward ``d_1`` (counter-clockwise when those axes are
    displayed as the usual ``(row, column)`` pair).

    Parameters
    ----------
    x : Tensor
        Input tensor (any rank :math:`\geq 2`).
    k : int, optional
        Number of 90° rotations to apply. Negative values rotate in the
        opposite direction. Reduced modulo 4 — values outside
        ``{0, 1, 2, 3}`` are equivalent to one of those four. Defaults
        to ``1``.
    dims : Sequence[int], optional
        Pair of axes defining the rotation plane. Defaults to ``(0, 1)``.

    Returns
    -------
    Tensor
        Tensor of the same rank as ``x``; the two axes in ``dims`` are
        permuted and one of them is flipped (their lengths swap when
        ``k`` is odd).

    Notes
    -----
    Implemented via flip + axis-swap:

    .. math::

        \text{rot90}_k(x) =
        \begin{cases}
            x,                                    & k \bmod 4 = 0, \\
            \operatorname{swap}_{d_0, d_1}(\operatorname{flip}_{d_1}(x)), & k \bmod 4 = 1, \\
            \operatorname{flip}_{d_0, d_1}(x),    & k \bmod 4 = 2, \\
            \operatorname{swap}_{d_0, d_1}(\operatorname{flip}_{d_0}(x)), & k \bmod 4 = 3.
        \end{cases}

    Applying ``rot90`` four times returns the original tensor.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([[1., 2.], [3., 4.]])
    >>> lucid.rot90(x)
    Tensor([[2., 4.],
            [1., 3.]])
    """
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
