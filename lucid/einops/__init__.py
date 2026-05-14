from lucid._tensor.tensor import Tensor

"""
lucid.einops: einops-style tensor operations.

Wraps the C++ engine's einops module. Supports patterns like
'b h w -> b (h w)' for rearrange, 'b h w -> b w' for reduce, etc.
"""

from lucid._C.engine import einops as _C_einops
from lucid._dispatch import _unwrap, _wrap


def rearrange(tensor: Tensor, pattern: str, **axes_lengths: int) -> Tensor:
    r"""Rearrange axes of ``tensor`` according to an einops pattern.

    Performs arbitrary axis manipulation — transpose, reshape,
    composition, and decomposition — expressed in a single readable
    pattern string. The notation makes the intent explicit and removes
    the chain of ``transpose``/``view`` calls that obscure dimensionality
    transforms in raw tensor code.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to rearrange.
    pattern : str
        Einops pattern of the form ``"<lhs> -> <rhs>"``. Parentheses
        group axes for composition or decomposition.
    **axes_lengths : int
        Named axis sizes used to disambiguate decomposed groups.

    Returns
    -------
    Tensor
        The rearranged tensor.

    Notes
    -----
    The transformation is purely a reshape and permutation — no data is
    copied unless required by memory layout. Typical patterns include
    ``"b c h w -> b (h w) c"`` for flattening spatial dims into a token
    sequence, and ``"b (h w) c -> b c h w"`` for the inverse.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(2, 3, 4)
    >>> lucid.einops.rearrange(x, "b h w -> b (h w)").shape
    (2, 12)
    """
    return _wrap(_C_einops.rearrange(_unwrap(tensor), pattern, axes_lengths))


def reduce(
    tensor: Tensor,
    pattern: str,
    reduction: str,
    **axes_lengths: int,
) -> Tensor:
    r"""Reduce ``tensor`` along axes implied by an einops pattern.

    Generalises standard ``sum`` / ``mean`` / ``max`` / ``min`` /
    ``prod`` reductions to arbitrary axes selected by symbolic pattern.
    Any axis present on the left-hand side of ``->`` but absent on the
    right-hand side is collapsed by the chosen reduction.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to reduce.
    pattern : str
        Einops pattern such as ``"b h w -> b w"``.
    reduction : str
        Reduction operation; one of ``"sum"``, ``"mean"``, ``"max"``,
        ``"min"``, ``"prod"``.
    **axes_lengths : int
        Named axis sizes used to disambiguate decomposed groups.

    Returns
    -------
    Tensor
        The reduced tensor.

    Notes
    -----
    Conceptually equivalent to a permutation followed by a reduction
    along contiguous trailing axes. For ``reduction="mean"`` the result
    is

    .. math::

        y_{b w} = \frac{1}{H} \sum_{h=1}^{H} x_{b h w}.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(2, 3, 4)
    >>> lucid.einops.reduce(x, "b h w -> b w", "mean").shape
    (2, 4)
    """
    return _wrap(_C_einops.reduce(_unwrap(tensor), pattern, reduction, axes_lengths))


def repeat(tensor: Tensor, pattern: str, **axes_lengths: int) -> Tensor:
    r"""Repeat ``tensor`` along new or existing axes via an einops pattern.

    Provides a single notation for both tiling along existing axes and
    broadcasting into entirely new axes. The pattern's right-hand side
    enumerates the desired output axes; any axis appearing on the right
    but not on the left is introduced and must be given an explicit
    length through ``**axes_lengths``.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to repeat.
    pattern : str
        Einops pattern such as ``"b h w -> b h w c"`` (new axis ``c``).
    **axes_lengths : int
        Sizes for every new axis introduced by the pattern.

    Returns
    -------
    Tensor
        The repeated tensor.

    Notes
    -----
    For an axis :math:`c` of size :math:`C` introduced via ``repeat``,

    .. math::

        y_{b h w c} = x_{b h w}, \quad c = 1, \ldots, C.

    No data is materialised until reading — internally the new axis is
    expressed as a broadcast / stride manipulation when possible.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(2, 3, 4)
    >>> lucid.einops.repeat(x, "b h w -> b h w c", c=5).shape
    (2, 3, 4, 5)
    """
    return _wrap(_C_einops.repeat(_unwrap(tensor), pattern, axes_lengths))


def einsum(equation: str, *operands: Tensor) -> Tensor:
    r"""Evaluate an Einstein-summation expression over the given tensors.

    Implements the general tensor contraction operator: each input is
    indexed by labels declared in ``equation`` and the output retains
    only the labels appearing on the right-hand side. Repeated labels
    are summed; matching labels broadcast.

    Parameters
    ----------
    equation : str
        Einsum equation of the form ``"<lhs1>,<lhs2>,... -> <rhs>"``.
    *operands : Tensor
        Tensors corresponding to the comma-separated factors on the
        left-hand side.

    Returns
    -------
    Tensor
        Tensor whose axes correspond to the labels on the right-hand
        side of ``equation``.

    Notes
    -----
    Matrix multiplication is the canonical example: ``"ij,jk->ik"``
    realises

    .. math::

        C_{ik} = \sum_{j} A_{ij} \, B_{jk}.

    More generally, einsum expresses traces (``"ii->"``), batched
    contractions (``"bij,bjk->bik"``), and outer products
    (``"i,j->ij"``) within one consistent notation.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.randn(2, 3)
    >>> b = lucid.randn(3, 4)
    >>> lucid.einops.einsum("ij,jk->ik", a, b).shape
    (2, 4)
    """
    impl_list = [_unwrap(t) for t in operands]
    return _wrap(_C_einops.einsum(equation, impl_list))  # type: ignore[arg-type]


__all__ = ["rearrange", "reduce", "repeat", "einsum"]
