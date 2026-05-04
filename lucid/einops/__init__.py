from lucid._tensor.tensor import Tensor

"""
lucid.einops: einops-style tensor operations.

Wraps the C++ engine's einops module. Supports patterns like
'b h w -> b (h w)' for rearrange, 'b h w -> b w' for reduce, etc.
"""

from typing import TYPE_CHECKING, TYPE_CHECKING
from lucid._C.engine import einops as _C_einops
from lucid._dispatch import _unwrap, _wrap


def rearrange(tensor: Tensor, pattern: str, **axes_lengths: int) -> Tensor:
    """Rearrange dimensions according to pattern.

    Args:
        tensor:       Input tensor.
        pattern:      Einops-style pattern, e.g. 'b h w -> b (h w)'.
        **axes_lengths: Named axis sizes for decomposition/composition.

    Returns:
        Rearranged tensor.

    Example:
        >>> x = lucid.randn(2, 3, 4)
        >>> rearrange(x, 'b h w -> b (h w)').shape
        (2, 12)
    """
    return _wrap(_C_einops.rearrange(_unwrap(tensor), pattern, axes_lengths))


def reduce(
    tensor: Tensor,
    pattern: str,
    reduction: str,
    **axes_lengths: int,
) -> Tensor:
    """Reduce tensor along axes specified by pattern.

    Args:
        tensor:     Input tensor.
        pattern:    Einops-style pattern, e.g. 'b h w -> b w'.
        reduction:  Reduction operation: 'mean', 'sum', 'max', 'min'.
        **axes_lengths: Named axis sizes.

    Returns:
        Reduced tensor.

    Example:
        >>> x = lucid.randn(2, 3, 4)
        >>> reduce(x, 'b h w -> b w', 'mean').shape
        (2, 4)
    """
    return _wrap(_C_einops.reduce(_unwrap(tensor), pattern, reduction, axes_lengths))


def repeat(tensor: Tensor, pattern: str, **axes_lengths: int) -> Tensor:
    """Repeat/tile tensor along new or existing axes.

    Args:
        tensor:       Input tensor.
        pattern:      Einops-style pattern, e.g. 'b h w -> b h w c'.
        **axes_lengths: Named axis sizes for new axes.

    Returns:
        Repeated tensor.

    Example:
        >>> x = lucid.randn(2, 3, 4)
        >>> repeat(x, 'b h w -> b h w c', c=5).shape
        (2, 3, 4, 5)
    """
    return _wrap(_C_einops.repeat(_unwrap(tensor), pattern, axes_lengths))


def einsum(equation: str, *operands: Tensor) -> Tensor:
    """Einstein summation on a sequence of tensors.

    Args:
        equation:  Einsum equation, e.g. 'ij,jk->ik'.
        *operands: Input tensors.

    Returns:
        Result tensor.

    Example:
        >>> a = lucid.randn(2, 3)
        >>> b = lucid.randn(3, 4)
        >>> einsum('ij,jk->ik', a, b).shape
        (2, 4)
    """
    impl_list = [_unwrap(t) for t in operands]
    return _wrap(_C_einops.einsum(equation, impl_list))


__all__ = ["rearrange", "reduce", "repeat", "einsum"]
