"""
nn.functional linear (fully-connected) operations.

Includes fused forward kernels (Phase 19 FusionPass):
  ``fused_linear_relu``  — y = max(0, x @ W.T + b)   (BLAS + vDSP, single pass)
  ``fused_linear_gelu``  — y = GELU(x @ W.T + b)     (BLAS + vForce tanh, single pass)

Both kernels bypass the standard two-op graph (matmul → activation) in
**inference mode** (``no_grad`` context or when no inputs require grad).
In training mode they transparently fall back to the unfused path so
autograd gradients remain correct.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """
    Apply a linear transformation: y = x @ weight.T + bias.

    Args:
        x:      (..., in_features)
        weight: (out_features, in_features)
        bias:   (out_features,) or None
    """
    w_impl = _unwrap(weight)
    out_features = w_impl.shape[0]
    if bias is not None:
        b_impl = _unwrap(bias)
    else:
        b_impl = _C_engine.zeros([out_features], w_impl.dtype, w_impl.device)
    return _wrap(_C_engine.nn.linear(_unwrap(x), w_impl, b_impl))


def bilinear(
    x1: Tensor,
    x2: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    r"""Bilinear transformation applied to two inputs.

    Computes a quadratic interaction between every pair of features from
    ``x1`` and ``x2``.  Common in fine-grained recognition (bilinear
    pooling), tensor-decomposition attention, and gated cross-modal
    fusion layers.

    Parameters
    ----------
    x1 : Tensor
        First input, shape ``(..., in1_features)``.
    x2 : Tensor
        Second input, shape ``(..., in2_features)``.
    weight : Tensor
        Weight tensor, shape ``(out_features, in1_features, in2_features)``.
    bias : Tensor, optional
        Bias vector, shape ``(out_features,)``.  Default ``None``.

    Returns
    -------
    Tensor
        Output tensor, shape ``(..., out_features)``.

    Notes
    -----
    For each output channel :math:`k`,

    .. math::

        y_k = x_1^\top W_k x_2 + b_k,

    where :math:`W_k` is the :math:`k`-th slice of the weight tensor.
    The parameter count grows multiplicatively in
    :math:`\text{in1} \times \text{in2}`, so bilinear layers are usually
    kept narrow.  Reduces to two ordinary linear layers when
    :math:`W_k = u_k v_k^\top` is rank-one (the *factorised bilinear*
    trick).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import bilinear
    >>> x1 = lucid.randn(2, 3)
    >>> x2 = lucid.randn(2, 4)
    >>> w = lucid.randn(5, 3, 4)
    >>> bilinear(x1, x2, w).shape
    (2, 5)
    """
    result = _wrap(
        _C_engine.nn.bilinear_layer(_unwrap(x1), _unwrap(x2), _unwrap(weight))
    )
    if bias is not None:
        result = _wrap(_C_engine.add(result._impl, _unwrap(bias)))
    return result


# ── Phase 19: Fused forward kernels ──────────────────────────────────────────


def fused_linear_relu(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    r"""Fused linear + ReLU forward kernel.

    Computes :math:`\text{ReLU}(xW^\top + b)` in a single CPU pass — the
    pre-activation tensor is never materialised, halving memory traffic on
    large MLP blocks.  Falls back to the unfused two-op path whenever any
    input requires gradient so that autograd remains correct.

    Parameters
    ----------
    x : Tensor
        Input, shape ``(..., in_features)``.
    weight : Tensor
        Weight matrix, shape ``(out_features, in_features)``.
    bias : Tensor
        Bias vector, shape ``(out_features,)``.

    Returns
    -------
    Tensor
        ``relu(x @ weight.T + bias)``, shape ``(..., out_features)``.

    Notes
    -----
    .. math::

        y = \max\!\big(0,\, x W^\top + b\big)

    Implementation: a single BLAS SGEMM followed by vDSP ``vrelu`` over
    the GEMM output buffer.  Selected by the Phase-19 FusionPass when
    grad-mode is disabled and none of ``x`` / ``weight`` / ``bias`` has
    ``requires_grad=True``.  In training mode the call routes through
    the standard :func:`linear` + :func:`relu` graph so autograd's
    backward derivations are exact.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import fused_linear_relu
    >>> x = lucid.randn(8, 16)
    >>> w = lucid.randn(32, 16)
    >>> b = lucid.zeros(32)
    >>> fused_linear_relu(x, w, b).shape
    (8, 32)
    """
    needs_grad: bool = _C_engine.grad_enabled() and (
        x.requires_grad or weight.requires_grad or bias.requires_grad
    )
    if needs_grad:
        # Training path: unfused standard ops — autograd graph is correct.
        return _wrap(
            _C_engine.relu(
                _C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias))
            )
        )

    # Inference path: single-pass BLAS+vDSP fused kernel.
    return _wrap(
        _C_engine._fused_linear_relu(_unwrap(x), _unwrap(weight), _unwrap(bias))
    )


def fused_linear_gelu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    approximate: str = "tanh",
) -> Tensor:
    r"""Fused linear + GELU forward kernel.

    Computes :math:`\text{GELU}(xW^\top + b)` in a single CPU pass.  Used
    extensively in transformer feed-forward blocks where the linear-then-
    GELU pattern dominates inference cost; fusing eliminates the
    intermediate activation buffer and improves cache behaviour.

    Parameters
    ----------
    x : Tensor
        Input, shape ``(..., in_features)``.
    weight : Tensor
        Weight matrix, shape ``(out_features, in_features)``.
    bias : Tensor
        Bias vector, shape ``(out_features,)``.
    approximate : str, optional
        GELU approximation.  ``"tanh"`` (default) is what the fused
        kernel implements; ``"none"`` falls back to the unfused exact-erf
        path.

    Returns
    -------
    Tensor
        ``gelu(x @ weight.T + bias)``, shape ``(..., out_features)``.

    Notes
    -----
    .. math::

        y = \text{GELU}\!\big(x W^\top + b\big)

    Implementation: a single BLAS SGEMM followed by vForce ``vtanh``
    over the GEMM output to evaluate the tanh-form GELU
    :math:`\tfrac{x}{2}\big[1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715 x^3))\big]`.
    Engaged only in inference mode (no grad-tracking inputs) and only
    when ``approximate="tanh"``; otherwise the call falls back to the
    standard :func:`linear` + :func:`gelu` graph so gradients remain
    exact.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import fused_linear_gelu
    >>> x = lucid.randn(8, 16)
    >>> w = lucid.randn(32, 16)
    >>> b = lucid.zeros(32)
    >>> fused_linear_gelu(x, w, b).shape
    (8, 32)
    """
    from lucid.nn.functional.activations import gelu as _gelu

    needs_grad: bool = _C_engine.grad_enabled() and (
        x.requires_grad or weight.requires_grad or bias.requires_grad
    )
    if needs_grad:
        # Training path: standard ops — autograd graph is correct.
        pre = _wrap(_C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias)))
        return _gelu(pre, approximate=approximate)

    if approximate == "tanh":
        return _wrap(
            _C_engine._fused_linear_gelu(_unwrap(x), _unwrap(weight), _unwrap(bias))
        )
    # exact erf path — no fused kernel available, fall back to two-op path.
    pre = _wrap(_C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias)))
    return _gelu(pre, approximate="none")
