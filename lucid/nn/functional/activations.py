"""
nn.functional activation functions.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def relu(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the rectified linear unit function element-wise.

    :math:`\\text{ReLU}(x) = \\max(0, x)`

    Parameters
    ----------
    x : Tensor
        Input tensor.
    inplace : bool, optional
        Currently ignored (not supported).

    Returns
    -------
    Tensor
        Tensor with negative values zeroed.

    Examples
    --------
    >>> F.relu(lucid.tensor([-1.0, 0.0, 1.0]))
    tensor([0., 0., 1.])
    """
    return _wrap(_C_engine.relu(_unwrap(x)))


def leaky_relu(
    x: Tensor, negative_slope: float = 0.01, inplace: bool = False
) -> Tensor:
    """Leaky rectified linear unit."""
    return _wrap(_C_engine.leaky_relu(_unwrap(x), negative_slope))


def elu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """Exponential linear unit."""
    return _wrap(_C_engine.elu(_unwrap(x), alpha))


def selu(x: Tensor, inplace: bool = False) -> Tensor:
    """Scaled exponential linear unit."""
    return _wrap(_C_engine.selu(_unwrap(x)))


def _erf_approx(xi: _C_engine.TensorImpl) -> _C_engine.TensorImpl:
    """Polynomial approximation of erf (Abramowitz & Stegun 7.1.26, max err < 1.5e-7)."""
    p_coef = 0.3275911
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    shape = list(xi.shape)
    dt, dev = xi.dtype, xi.device

    abs_xi = _C_engine.abs(xi)
    ones = _C_engine.ones(shape, dt, dev)
    # t = 1 / (1 + p * |x|)
    t = _C_engine.div(
        ones,
        _C_engine.add(
            ones, _C_engine.mul(_C_engine.full(shape, p_coef, dt, dev), abs_xi)
        ),
    )
    # Horner: ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
    poly = _C_engine.add(
        _C_engine.mul(_C_engine.full(shape, a5, dt, dev), t),
        _C_engine.full(shape, a4, dt, dev),
    )
    poly = _C_engine.add(_C_engine.mul(poly, t), _C_engine.full(shape, a3, dt, dev))
    poly = _C_engine.add(_C_engine.mul(poly, t), _C_engine.full(shape, a2, dt, dev))
    poly = _C_engine.add(_C_engine.mul(poly, t), _C_engine.full(shape, a1, dt, dev))
    poly = _C_engine.mul(poly, t)
    # erf(|x|) = 1 - poly * exp(-x^2)
    erf_abs = _C_engine.sub(
        ones,
        _C_engine.mul(
            poly, _C_engine.exp(_C_engine.neg(_C_engine.mul(abs_xi, abs_xi)))
        ),
    )
    # Restore sign: erf(x) = sign(x) * erf(|x|), with erf(0)=0 handled by sign(0)=0
    return _C_engine.where(
        _C_engine.equal(xi, _C_engine.zeros(shape, dt, dev)),
        _C_engine.zeros(shape, dt, dev),
        _C_engine.mul(_C_engine.sign(xi), erf_abs),
    )


def gelu(x: Tensor, approximate: str = "none") -> Tensor:
    """Gaussian error linear unit.

    approximate="none"  (default) → exact erf-based formula (matches reference default).
    approximate="tanh"            → tanh-approximation (faster, slightly less accurate).
    """
    xi = _unwrap(x)
    if approximate == "tanh":
        return _wrap(_C_engine.gelu(xi))
    # exact: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2_inv = 0.7071067811865476
    shape = list(xi.shape)
    dt, dev = xi.dtype, xi.device
    x_scaled = _C_engine.mul(xi, _C_engine.full(shape, sqrt2_inv, dt, dev))
    erf_val = _erf_approx(x_scaled)
    half = _C_engine.full(shape, 0.5, dt, dev)
    ones = _C_engine.ones(shape, dt, dev)
    return _wrap(_C_engine.mul(_C_engine.mul(xi, half), _C_engine.add(ones, erf_val)))


def silu(x: Tensor, inplace: bool = False) -> Tensor:
    """Sigmoid linear unit (Swish)."""
    return _wrap(_C_engine.silu(_unwrap(x)))


def mish(x: Tensor) -> Tensor:
    """Mish activation."""
    return _wrap(_C_engine.mish(_unwrap(x)))


def hardswish(x: Tensor) -> Tensor:
    """Hard Swish activation."""
    return _wrap(_C_engine.hard_swish(_unwrap(x)))


def hardsigmoid(x: Tensor) -> Tensor:
    """Hard sigmoid activation."""
    return _wrap(_C_engine.hard_sigmoid(_unwrap(x)))


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation."""
    return _wrap(_C_engine.sigmoid(_unwrap(x)))


def tanh(x: Tensor) -> Tensor:
    """Hyperbolic tangent."""
    return _wrap(_C_engine.tanh(_unwrap(x)))


def softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Apply softmax along a dimension.

    :math:`\\text{Softmax}(x_i) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}`

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int, optional
        Dimension along which softmax is computed. Defaults to ``-1``.

    Returns
    -------
    Tensor
        Tensor of the same shape with values summing to 1 along ``dim``.

    Examples
    --------
    >>> logits = lucid.tensor([[1.0, 2.0, 3.0]])
    >>> F.softmax(logits, dim=1).sum()   # ≈ 1.0
    """
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_unwrap(x), axis))


def log_softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Log-softmax along dim (numerically stable)."""
    axis = dim if dim is not None else -1
    sm = _C_engine.softmax(_unwrap(x), axis)
    return _wrap(_C_engine.log(sm))


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Softplus activation: ``(1/beta) * log(1 + exp(beta * x))``.

    Falls back to the identity ``x`` element-wise when ``beta * x > threshold``
    so the ``exp`` term doesn't overflow on large positive inputs — matches
    the reference framework's ``F.softplus`` semantics.

    The engine's bare ``softplus`` kernel implements the special case
    ``beta=1, threshold≈inf``; when either argument deviates we compose the
    full formula via ``mul``/``log``/``exp``/``where`` so the gradient still
    flows through the existing backward nodes.
    """
    if beta == 1.0 and threshold >= 50.0:
        # Hot-path: the bare engine kernel is already correct here.
        return _wrap(_C_engine.softplus(_unwrap(x)))
    xi = _unwrap(x)
    beta_t = _C_engine.full(list(xi.shape), float(beta), xi.dtype, xi.device)
    threshold_t = _C_engine.full(list(xi.shape), float(threshold), xi.dtype, xi.device)
    bx = _C_engine.mul(xi, beta_t)
    # log1p(exp(bx)) / beta — composed because the engine's ``softplus`` is
    # beta-agnostic.
    inv_beta = _C_engine.full(list(xi.shape), 1.0 / float(beta), xi.dtype, xi.device)
    softplus_full = _C_engine.mul(_C_engine.softplus(bx), inv_beta)
    above = _C_engine.greater(bx, threshold_t)
    return _wrap(_C_engine.where(above, xi, softplus_full))


def relu6(x: Tensor, inplace: bool = False) -> Tensor:
    """ReLU6 activation (clamped at 6)."""
    return _wrap(_C_engine.relu6(_unwrap(x)))


def softmin(x: Tensor, dim: int | None = None) -> Tensor:
    """Softmin: softmax applied to the negation of x."""
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_C_engine.neg(_unwrap(x)), axis))


def glu(x: Tensor, dim: int = -1) -> Tensor:
    """Gated linear unit: splits x along dim, returns first * sigmoid(second)."""
    impl = _unwrap(x)
    n = impl.shape[dim] // 2
    # Split into two halves along dim
    parts = _C_engine.split_at(impl, [n], dim)
    first, second = parts[0], parts[1]
    return _wrap(_C_engine.mul(first, _C_engine.sigmoid(second)))


def prelu(x: Tensor, weight: Tensor) -> Tensor:
    """Parametric ReLU: max(0,x) + weight * min(0,x)."""
    xi = _unwrap(x)
    wi = _unwrap(weight)
    pos = _C_engine.relu(xi)
    neg_part = _C_engine.mul(
        wi, _C_engine.minimum(_C_engine.zeros(xi.shape, xi.dtype, xi.device), xi)
    )
    return _wrap(_C_engine.add(pos, neg_part))


def celu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """Continuously differentiable ELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1))."""
    xi = _unwrap(x)
    pos = _C_engine.relu(xi)
    exp_part = _C_engine.sub(
        _C_engine.exp(
            _C_engine.div(xi, _C_engine.full(xi.shape, alpha, xi.dtype, xi.device))
        ),
        _C_engine.full(xi.shape, 1.0, xi.dtype, xi.device),
    )
    neg = _C_engine.minimum(
        _C_engine.zeros(xi.shape, xi.dtype, xi.device),
        _C_engine.mul(_C_engine.full(xi.shape, alpha, xi.dtype, xi.device), exp_part),
    )
    return _wrap(_C_engine.add(pos, neg))


def hardshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    """Hard shrinkage: x if |x|>lambd else 0."""
    xi = _unwrap(x)
    lam = _C_engine.full(xi.shape, lambd, xi.dtype, xi.device)
    neg_lam = _C_engine.full(xi.shape, -lambd, xi.dtype, xi.device)
    mask = _C_engine.bitwise_or(
        _C_engine.greater(xi, lam),
        _C_engine.less(xi, neg_lam),
    )
    return _wrap(
        _C_engine.where(mask, xi, _C_engine.zeros(xi.shape, xi.dtype, xi.device))
    )


def tanhshrink(x: Tensor) -> Tensor:
    """Tanh shrinkage: x - tanh(x)."""
    xi = _unwrap(x)
    return _wrap(_C_engine.sub(xi, _C_engine.tanh(xi)))


def normalize(
    x: Tensor,
    p: float = 2.0,
    dim: int = 1,
    eps: float = 1e-12,
) -> Tensor:
    """Normalize a tensor to unit :math:`L_p` norm along a dimension.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    p : float, optional
        Exponent of the norm (default: 2.0 for Euclidean norm).
    dim : int, optional
        Dimension along which to normalize (default: 1).
    eps : float, optional
        Small value added to the denominator for numerical stability.

    Returns
    -------
    Tensor
        Normalized tensor with unit :math:`L_p` norm along ``dim``.

    Examples
    --------
    >>> x = lucid.tensor([[3.0, 4.0]])
    >>> F.normalize(x, p=2, dim=1)
    tensor([[0.6, 0.8]])
    """
    return _wrap(_C_engine.nn.lp_normalize(_unwrap(x), p, dim, eps))


def cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps: float = 1e-8,
) -> Tensor:
    """Compute cosine similarity along dim."""
    x1n = normalize(x1, p=2.0, dim=dim, eps=eps)
    x2n = normalize(x2, p=2.0, dim=dim, eps=eps)
    # Element-wise product then sum along dim
    impl = _C_engine.sum(_C_engine.mul(_unwrap(x1n), _unwrap(x2n)), [dim])
    return _wrap(impl)


def pairwise_distance(
    x1: Tensor,
    x2: Tensor,
    p: float = 2.0,
    eps: float = 1e-6,
    keepdim: bool = False,
) -> Tensor:
    """Compute pairwise L_p distance between x1 and x2."""
    diff = _C_engine.sub(_unwrap(x1), _unwrap(x2))
    # |diff|_p = (sum |diff|^p)^(1/p)
    abs_diff = _C_engine.abs(diff)
    powered = _C_engine.pow(
        _C_engine.add(
            abs_diff,
            _C_engine.full(abs_diff.shape, eps, abs_diff.dtype, abs_diff.device),
        ),
        _C_engine.full(abs_diff.shape, p, abs_diff.dtype, abs_diff.device),
    )
    ndim = len(powered.shape)
    last_dim = ndim - 1
    s = _C_engine.sum(powered, [last_dim], False)
    inv_p = _C_engine.full(s.shape, 1.0 / p, s.dtype, s.device)
    dist = _C_engine.pow(s, inv_p)
    if keepdim:
        dist = _C_engine.unsqueeze(dist, last_dim)
    return _wrap(dist)


def softshrink(x: "Tensor", lambd: float = 0.5) -> "Tensor":
    """Soft-shrinkage: x-lambd if x>lambd, x+lambd if x<-lambd, else 0."""
    xi = _unwrap(x)
    lam = _C_engine.full(xi.shape, lambd, xi.dtype, xi.device)
    neg_lam = _C_engine.full(xi.shape, -lambd, xi.dtype, xi.device)
    zeros = _C_engine.zeros(xi.shape, xi.dtype, xi.device)
    pos_mask = _C_engine.greater(xi, lam)
    neg_mask = _C_engine.less(xi, neg_lam)
    pos_val = _C_engine.sub(xi, lam)
    neg_val = _C_engine.add(xi, lam)
    out = _C_engine.where(pos_mask, pos_val, zeros)
    out = _C_engine.where(neg_mask, neg_val, out)
    return _wrap(out)
