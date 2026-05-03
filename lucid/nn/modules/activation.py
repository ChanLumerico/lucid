"""
Activation function modules.
"""

from typing import Any
from lucid.nn.module import Module
# F imported lazily inside forward()


class ReLU(Module):
    """Rectified linear unit."""
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.relu(x, self.inplace)
    def extra_repr(self) -> str:
        return f"inplace={self.inplace}" if self.inplace else ""

class LeakyReLU(Module):
    """Leaky rectified linear unit."""
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.leaky_relu(x, self.negative_slope, self.inplace)
    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"

class ELU(Module):
    """Exponential linear unit."""
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.elu(x, self.alpha, self.inplace)
    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"

class SELU(Module):
    """Scaled exponential linear unit."""
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.selu(x, self.inplace)

class GELU(Module):
    """Gaussian error linear unit."""
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.gelu(x, self.approximate)
    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}" if self.approximate != "none" else ""

class SiLU(Module):
    """Sigmoid linear unit (Swish)."""
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.silu(x)

class Mish(Module):
    """Mish activation."""
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.mish(x)

class Hardswish(Module):
    """Hard Swish activation."""
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.hardswish(x)

class Hardsigmoid(Module):
    """Hard sigmoid activation."""
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.hardsigmoid(x)

class Sigmoid(Module):
    """Sigmoid activation."""
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.sigmoid(x)

class Tanh(Module):
    """Hyperbolic tangent."""
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.tanh(x)

class Softmax(Module):
    """Softmax activation."""
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.softmax(x, self.dim)
    def extra_repr(self) -> str:
        return f"dim={self.dim}"

class LogSoftmax(Module):
    """Log-softmax activation."""
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.log_softmax(x, self.dim)
    def extra_repr(self) -> str:
        return f"dim={self.dim}"

class ReLU6(Module):
    """ReLU6 activation."""
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.relu6(x)
