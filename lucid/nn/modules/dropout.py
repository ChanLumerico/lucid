"""
Dropout modules.
"""

from typing import Any
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.dropout import dropout, dropout2d


class Dropout(Module):
    """Randomly zero elements during training."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Any) -> Any:
        return dropout(x, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Dropout2d(Module):
    """Randomly zero entire channels during training."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Any) -> Any:
        return dropout2d(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class AlphaDropout(Module):
    """Alpha dropout for SELU networks."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Any) -> Any:
        return _wrap(_C_engine.nn.alpha_dropout(_unwrap(x), self.p, self.training))

    def extra_repr(self) -> str:
        return f"p={self.p}"
