"""
Activation function modules.
"""

from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import full
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.activations import (
    relu,
    leaky_relu,
    elu,
    selu,
    gelu,
    silu,
    mish,
    hardswish,
    hardsigmoid,
    sigmoid,
    tanh,
    softmax,
    log_softmax,
    relu6,
    prelu,
    softmin,
    glu,
)


class ReLU(Module):
    """Rectified linear unit."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Any) -> Any:
        return relu(x, self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}" if self.inplace else ""


class LeakyReLU(Module):
    """Leaky rectified linear unit."""

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: Any) -> Any:
        return leaky_relu(x, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class ELU(Module):
    """Exponential linear unit."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x: Any) -> Any:
        return elu(x, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class SELU(Module):
    """Scaled exponential linear unit."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Any) -> Any:
        return selu(x, self.inplace)


class GELU(Module):
    """Gaussian error linear unit."""

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Any) -> Any:
        return gelu(x, self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}" if self.approximate != "none" else ""


class SiLU(Module):
    """Sigmoid linear unit (Swish)."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()

    def forward(self, x: Any) -> Any:
        return silu(x)


class Mish(Module):
    """Mish activation."""

    def forward(self, x: Any) -> Any:
        return mish(x)


class Hardswish(Module):
    """Hard Swish activation."""

    def forward(self, x: Any) -> Any:
        return hardswish(x)


class Hardsigmoid(Module):
    """Hard sigmoid activation."""

    def forward(self, x: Any) -> Any:
        return hardsigmoid(x)


class Sigmoid(Module):
    """Sigmoid activation."""

    def forward(self, x: Any) -> Any:
        return sigmoid(x)


class Tanh(Module):
    """Hyperbolic tangent."""

    def forward(self, x: Any) -> Any:
        return tanh(x)


class Softmax(Module):
    """Softmax activation."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Any) -> Any:
        return softmax(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSoftmax(Module):
    """Log-softmax activation."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Any) -> Any:
        return log_softmax(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class ReLU6(Module):
    """ReLU6 activation."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()

    def forward(self, x: Any) -> Any:
        return relu6(x)


class PReLU(Module):
    """Parametric rectified linear unit with learnable slope."""

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(
            full((num_parameters,), init, dtype=dtype, device=device)
        )

    def forward(self, x: Any) -> Any:
        return prelu(x, self.weight)

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}"


class Threshold(Module):
    """Threshold activation: y = x if x > threshold, else value."""

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x: Any) -> Any:
        impl = _unwrap(x)
        fill = _C_engine.full(impl.shape, self.value, impl.dtype, impl.device)
        thresh = _C_engine.full(impl.shape, self.threshold, impl.dtype, impl.device)
        mask = _C_engine.greater(impl, thresh)
        return _wrap(_C_engine.where(mask, impl, fill))

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, value={self.value}"


class Hardtanh(Module):
    """Hardtanh: clamp to [min_val, max_val]."""

    def __init__(
        self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False
    ) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: Any) -> Any:
        impl = _unwrap(x)
        return _wrap(_C_engine.clip(impl, self.min_val, self.max_val))

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}, max_val={self.max_val}"


class LogSigmoid(Module):
    """Log-sigmoid: log(sigmoid(x))."""

    def forward(self, x: Any) -> Any:
        return _wrap(_C_engine.log(_C_engine.sigmoid(_unwrap(x))))


class Softsign(Module):
    """Softsign: x / (1 + |x|)."""

    def forward(self, x: Any) -> Any:
        impl = _unwrap(x)
        denom = _C_engine.add(
            _C_engine.full(impl.shape, 1.0, impl.dtype, impl.device),
            _C_engine.abs(impl),
        )
        return _wrap(_C_engine.div(impl, denom))


class Softmin(Module):
    """Softmin activation: softmax(-x)."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Any) -> Any:
        return softmin(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class GLU(Module):
    """Gated linear unit along dim."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Any) -> Any:
        return glu(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
