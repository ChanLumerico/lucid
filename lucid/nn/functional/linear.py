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
    """Bilinear transformation: y = x1 @ W @ x2.T + bias."""
    result = _wrap(
        _C_engine.nn.bilinear_layer(_unwrap(x1), _unwrap(x2), _unwrap(weight))
    )
    if bias is not None:
        result = _wrap(_C_engine.add(result._impl, _unwrap(bias)))
    return result


# ── Phase 19: Fused forward kernels ──────────────────────────────────────────


def fused_linear_relu(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Fused linear + ReLU: ``y = max(0, x @ weight.T + bias)``.

    In **inference mode** (no gradient tracking required) this executes as a
    single BLAS SGEMM + vDSP ``vrelu`` pass on CPU — avoiding the intermediate
    allocation of the pre-activation tensor.

    In **training mode** (any input has ``requires_grad=True``) it falls back
    to the standard unfused path so autograd can build a correct backward
    graph.

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
    """
    needs_grad: bool = (
        _C_engine.grad_enabled()
        and (x.requires_grad or weight.requires_grad or bias.requires_grad)
    )
    if needs_grad:
        # Training path: unfused standard ops — autograd graph is correct.
        return _wrap(_C_engine.relu(
            _C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias))
        ))

    # Inference path: single-pass BLAS+vDSP fused kernel.
    return _wrap(_C_engine._fused_linear_relu(_unwrap(x), _unwrap(weight), _unwrap(bias)))


def fused_linear_gelu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    approximate: str = "tanh",
) -> Tensor:
    """Fused linear + GELU: ``y = GELU(x @ weight.T + bias)``.

    In **inference mode** this executes as a single BLAS SGEMM + vForce
    ``vtanh`` pass (tanh approximation) on CPU.

    In **training mode** it falls back to the unfused path.

    Parameters
    ----------
    x : Tensor
        Input, shape ``(..., in_features)``.
    weight : Tensor
        Weight matrix, shape ``(out_features, in_features)``.
    bias : Tensor
        Bias vector, shape ``(out_features,)``.
    approximate : str
        GELU approximation.  ``'tanh'`` (default) matches the fused kernel.
        ``'none'`` uses the exact erf-based GELU on the unfused path.

    Returns
    -------
    Tensor
        ``gelu(x @ weight.T + bias)``, shape ``(..., out_features)``.
    """
    from lucid.nn.functional.activations import gelu as _gelu

    needs_grad: bool = (
        _C_engine.grad_enabled()
        and (x.requires_grad or weight.requires_grad or bias.requires_grad)
    )
    if needs_grad:
        # Training path: standard ops — autograd graph is correct.
        pre = _wrap(_C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias)))
        return _gelu(pre, approximate=approximate)

    if approximate == "tanh":
        return _wrap(_C_engine._fused_linear_gelu(_unwrap(x), _unwrap(weight), _unwrap(bias)))
    # exact erf path — no fused kernel available, fall back to two-op path.
    pre = _wrap(_C_engine.nn.linear(_unwrap(x), _unwrap(weight), _unwrap(bias)))
    return _gelu(pre, approximate="none")
