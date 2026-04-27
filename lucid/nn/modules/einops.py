"""
lucid.nn.modules.einops — module wrapper for einops `rearrange`.
"""

from __future__ import annotations

import lucid
import lucid.nn as nn

from lucid._tensor import Tensor


__all__ = ["Rearrange"]


@nn.auto_repr("pattern")
class Rearrange(nn.Module):
    def __init__(self, pattern: str, **shapes: int) -> None:
        super().__init__()
        self.pattern = pattern
        self.shapes = shapes

    def forward(self, input_: Tensor) -> Tensor:
        return lucid.einops.rearrange(input_, self.pattern, **self.shapes)
