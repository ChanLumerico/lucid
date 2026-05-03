"""
Linear and related fully-connected layers.
"""

import math
from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
# F imported lazily inside forward()
import lucid.nn.init as init
from lucid._factories.creation import empty


class Linear(Module):
    """
    Fully connected linear layer: y = x @ weight.T + bias.

    Args:
        in_features:  number of input features
        out_features: number of output features
        bias:         if True, add a learnable bias
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
        from lucid.nn import functional as F
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Identity(Module):
    """Pass-through layer that returns its input unchanged."""

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
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
        from lucid.nn import functional as F
        return F.bilinear(x1, x2, self.weight, self.bias)
