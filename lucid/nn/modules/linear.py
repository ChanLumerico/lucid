"""
Linear and related fully-connected layers.
"""

import math
from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
import lucid.nn.init as init
from lucid._factories.creation import empty
from lucid.nn.functional.linear import linear, bilinear


class Linear(Module):
    """Apply a linear transformation: :math:`y = xW^T + b`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If ``True`` (default), add a learnable bias term.
    device : optional
        Device for the initial parameters.
    dtype : optional
        Data type for the initial parameters.

    Attributes
    ----------
    weight : Parameter
        Shape ``(out_features, in_features)``. Initialized with Kaiming uniform.
    bias : Parameter or None
        Shape ``(out_features,)``. ``None`` when ``bias=False``.

    Examples
    --------
    >>> m = nn.Linear(20, 10)
    >>> x = lucid.randn(4, 20)
    >>> m(x).shape
    (4, 10)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias: Parameter | None = Parameter(empty(out_features, dtype=dtype, device=device))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight with Kaiming uniform and bias with uniform fan_in bound."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Any) -> Any:
        return linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Identity(Module):
    """Pass-through layer that returns its input unchanged."""

    def forward(self, x: Any) -> Any:
        return x


class Bilinear(Module):
    """Bilinear layer: y = x1 @ W @ x2.T + bias."""

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = Parameter(empty(out_features, in1_features, in2_features, dtype=dtype, device=device))
        self.bias: Parameter | None = (
            Parameter(empty(out_features, dtype=dtype, device=device)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize with Kaiming uniform."""
        import math
        bound = 1.0 / math.sqrt(self.weight.shape[1])
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x1: Any, x2: Any) -> Any:
        return bilinear(x1, x2, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None}")
