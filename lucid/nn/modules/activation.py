"""
Activation function modules.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import full
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.activations import (
    relu,
    leaky_relu,
    elu,
    celu,
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
    hardshrink,
    tanhshrink,
    softshrink,
    softplus,
    rrelu,
    cosine_similarity,
    pairwise_distance,
)


class ReLU(Module):
    """Rectified linear unit."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return relu(x, self.inplace)

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}" if self.inplace else ""


class LeakyReLU(Module):
    """Leaky rectified linear unit."""

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return leaky_relu(x, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class ELU(Module):
    """Exponential linear unit."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return elu(x, self.alpha, self.inplace)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class SELU(Module):
    """Scaled exponential linear unit."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return selu(x, self.inplace)


class GELU(Module):
    """Gaussian error linear unit."""

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x, self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}" if self.approximate != "none" else ""


class SiLU(Module):
    """Sigmoid linear unit (Swish)."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return silu(x)


class Mish(Module):
    """Mish activation."""

    def forward(self, x: Tensor) -> Tensor:
        return mish(x)


class Softplus(Module):
    """Smooth approximation of ReLU: ``log(1 + exp(beta * x)) / beta``.

    For numerical stability, ``softplus`` falls back to the identity
    ``x`` when ``beta * x > threshold`` — captures the asymptote without
    overflowing the ``exp`` term.
    """

    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        return softplus(x, self.beta, self.threshold)

    def extra_repr(self) -> str:
        if self.beta == 1.0 and self.threshold == 20.0:
            return ""
        return f"beta={self.beta}, threshold={self.threshold}"


class Hardswish(Module):
    """Hard Swish activation."""

    def forward(self, x: Tensor) -> Tensor:
        return hardswish(x)


class Hardsigmoid(Module):
    """Hard sigmoid activation."""

    def forward(self, x: Tensor) -> Tensor:
        return hardsigmoid(x)


class Sigmoid(Module):
    """Sigmoid activation."""

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Tanh(Module):
    """Hyperbolic tangent."""

    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)


class Softmax(Module):
    """Softmax activation."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSoftmax(Module):
    """Log-softmax activation."""

    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return log_softmax(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Softmax2d(Module):
    """Apply softmax over the channel dimension of a 4-D tensor.

    Shorthand for ``softmax(x, dim=-3)`` on ``(N, C, H, W)`` inputs —
    used by segmentation heads where each pixel's channel scores form a
    categorical distribution.  No ``dim`` parameter; the channel axis
    is always second-to-last (matches the reference framework).
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Softmax2d: expected 4-D input (N, C, H, W), got ndim={x.ndim}"
            )
        return softmax(x, -3)


class RReLU(Module):
    """Randomized leaky ReLU.

    Per-element negative slope drawn uniformly from ``[lower, upper]``
    during training; eval mode uses the fixed midpoint
    ``(lower + upper) / 2`` (the slope's expectation), matching the
    reference framework.
    """

    def __init__(
        self,
        lower: float = 1.0 / 8.0,
        upper: float = 1.0 / 3.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return rrelu(x, self.lower, self.upper, training=self.training, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"lower={self.lower}, upper={self.upper}"


class ReLU6(Module):
    """ReLU6 activation."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return relu6(x)


class PReLU(Module):
    """Parametric rectified linear unit with learnable slope."""

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(
            full((num_parameters,), init, dtype=dtype, device=device)
        )

    def forward(self, x: Tensor) -> Tensor:
        return prelu(x, self.weight)

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}"


class Threshold(Module):
    """Threshold activation: y = x if x > threshold, else value."""

    def __init__(self, threshold: float, value: float, inplace: bool = False) -> None:
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        impl = _unwrap(x)
        return _wrap(_C_engine.clip(impl, self.min_val, self.max_val))

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}, max_val={self.max_val}"


class LogSigmoid(Module):
    """Log-sigmoid: log(sigmoid(x))."""

    def forward(self, x: Tensor) -> Tensor:
        return _wrap(_C_engine.log(_C_engine.sigmoid(_unwrap(x))))


class Softsign(Module):
    """Softsign: x / (1 + |x|)."""

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        return softmin(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class GLU(Module):
    """Gated linear unit along dim."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return glu(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class CELU(Module):
    """Continuously differentiable ELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1))."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return celu(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class Hardshrink(Module):
    """Hard shrinkage: x if |x|>lambda else 0."""

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return hardshrink(x, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


class Tanhshrink(Module):
    """Tanhshrink: x - tanh(x)."""

    def forward(self, x: Tensor) -> Tensor:
        return tanhshrink(x)


class Softshrink(Module):
    """Softshrink: x - lambd if x > lambd; x + lambd if x < -lambd; 0 otherwise."""

    def __init__(self, lambd: float = 0.5) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        return softshrink(x, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


# ── distance / similarity wrappers ─────────────────────────────────────────


class CosineSimilarity(Module):
    """Module wrapping :func:`cosine_similarity`.

    Returns the cosine of the angle between ``x1`` and ``x2`` along
    ``dim``, with a small ``eps`` added to the magnitudes for numerical
    stability."""

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


class PairwiseDistance(Module):
    """Module wrapping :func:`pairwise_distance` — element-wise L_p
    distance between rows of two tensors."""

    def __init__(
        self,
        p: float = 2.0,
        eps: float = 1e-6,
        keepdim: bool = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pairwise_distance(x1, x2, p=self.p, eps=self.eps, keepdim=self.keepdim)

    def extra_repr(self) -> str:
        return f"p={self.p}, eps={self.eps}, keepdim={self.keepdim}"
