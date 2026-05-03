"""
nn.functional sparse / embedding operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def embedding(
    x: "Tensor",
    weight: "Tensor",
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> "Tensor":
    """Look up rows in weight by integer indices x."""
    return _wrap(_C_engine.nn.embedding(_unwrap(x), _unwrap(weight)))


def one_hot(tensor: "Tensor", num_classes: int = -1) -> "Tensor":
    """Convert integer class indices to one-hot encoded tensor."""
    return _wrap(_C_engine.nn.one_hot(_unwrap(tensor), num_classes))
