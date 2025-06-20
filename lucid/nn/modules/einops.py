import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.einops._func import _EinopsPattern


__all__ = ["Rearrange"]


@nn.auto_repr("pattern")
class Rearrange(nn.Module):
    def __init__(self, pattern: _EinopsPattern, **shapes: int) -> None:
        super().__init__()
        self.pattern = pattern
        self.shapes = shapes

    def forward(self, input_: Tensor) -> Tensor:
        return lucid.einops.rearrange(input_, self.pattern, **self.shapes)
