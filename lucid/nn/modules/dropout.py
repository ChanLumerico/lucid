"""
Dropout modules.
"""

from typing import Any
from lucid.nn.module import Module
# F imported lazily inside forward()


class Dropout(Module):
    """Randomly zero elements during training."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.dropout(x, self.p, self.training, self.inplace)


class Dropout2d(Module):
    """Randomly zero entire channels during training."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.dropout2d(x, self.p, self.training)


class AlphaDropout(Module):
    """Alpha dropout for SELU networks."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _unwrap, _wrap
        return _wrap(_C_engine.nn.alpha_dropout(_unwrap(x), self.p, self.training))
