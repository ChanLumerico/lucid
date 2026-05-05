"""
Dropout modules.
"""

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.dropout import dropout, dropout2d


def _check_dropout_prob(p: float) -> None:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {p!r}")


class Dropout(Module):
    """Randomly zero individual elements during training.

    During training, each element of the input is zeroed independently
    with probability ``p``; the surviving elements are scaled by
    ``1/(1-p)`` so the expected value is preserved.  In eval mode the
    layer is the identity.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return dropout(x, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class Dropout2d(Module):
    """Randomly zero entire channels during training (4-D input).

    Equivalent to a per-channel Bernoulli mask broadcast over the spatial
    dimensions — useful for adjacent feature maps that are spatially
    correlated, where elementwise dropout is too local.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return dropout2d(x, self.p, self.training)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class AlphaDropout(Module):
    """Alpha dropout — preserves the SELU mean/variance contract."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return _wrap(_C_engine.nn.alpha_dropout(_unwrap(x), self.p, self.training))

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class Dropout3d(Module):
    """Randomly zeros entire 3-D feature maps (channels) for 5-D input."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: "Tensor") -> "Tensor":
        from lucid.nn.functional.dropout import dropout3d

        return dropout3d(x, self.p, self.training)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s
