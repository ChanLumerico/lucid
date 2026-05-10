"""
Dropout modules.
"""

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.dropout import dropout, dropout2d, feature_alpha_dropout


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

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return dropout(x, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class Dropout1d(Module):
    """Randomly zero entire channels during training (3-D input).

    Same channel-wise mask semantics as ``Dropout2d`` but for inputs shaped
    ``(N, C, L)`` — the per-channel Bernoulli mask is broadcast across the
    length dimension.  Useful for 1-D conv stacks where adjacent positions
    along ``L`` are correlated and elementwise dropout would be too local.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        # The engine's ``dropoutnd`` kernel handles 3-D / 4-D / 5-D inputs by
        # building the mask along the channel axis, so the same call works
        # here as for ``Dropout2d``.
        return dropout2d(x, self.p, self.training)

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

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
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

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
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

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        from lucid.nn.functional.dropout import dropout3d

        return dropout3d(x, self.p, self.training)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s


class FeatureAlphaDropout(Module):
    """Alpha dropout applied per-channel (whole feature maps).

    Like ``AlphaDropout`` but the Bernoulli mask is sampled along the
    ``(N, C)`` axes and broadcast over the spatial dims — preserves the
    self-normalising property of SELU networks while encouraging
    feature-map-level regularisation.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        _check_dropout_prob(p)
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return feature_alpha_dropout(x, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        s: str = f"p={self.p}"
        if self.inplace:
            s += ", inplace=True"
        return s
