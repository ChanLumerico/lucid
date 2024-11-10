"""
`lucid.random`
--------------
Lucid's random number generation package.
"""

from lucid.random import _func

from lucid._tensor import Tensor
from lucid.types import _ShapeLike


def seed(seed: int) -> None:
    """Set a global random seed."""
    return _func.seed(seed)


def rand(shape: _ShapeLike, requires_grad: bool = False) -> Tensor:
    """Create a tensor of the given shape from an uniform distribution `[0, 1)`."""
    return _func.rand(shape, requires_grad)


def randint(
    low: int, high: int | None, size: int | _ShapeLike, requires_grad: bool = False
) -> Tensor:
    """Create a tensor of the given size from bounded integer range."""
    return _func.randint(low, high, size, requires_grad)


def randn(shape: _ShapeLike, requires_grad: bool = False) -> Tensor:
    """Create a tensor of the given shape from 'standard normal' distribution."""
    return _func.randn(shape, requires_grad)
