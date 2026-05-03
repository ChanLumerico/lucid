"""
nn.functional dropout operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def dropout(
    x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    """Randomly zero elements with probability p during training."""
    return _wrap(_C_engine.nn.dropout(_unwrap(x), p, training))


def dropout2d(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Randomly zero entire channels with probability p during training."""
    return _wrap(_C_engine.nn.dropoutnd(_unwrap(x), p, training))
