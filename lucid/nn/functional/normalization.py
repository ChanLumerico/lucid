"""
nn.functional normalization operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def batch_norm(
    x: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Batch normalization.

    The engine implements ``(x, gamma, beta, eps)``.  running_mean / running_var
    are tracked in Python but not forwarded to the engine (which normalises using
    the batch statistics unconditionally).
    """
    from lucid._factories.creation import ones, zeros
    C = x.shape[1]
    w = _unwrap(weight) if weight is not None else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    b = _unwrap(bias) if bias is not None else _unwrap(zeros(C, device=x.device, dtype=x.dtype))
    return _wrap(_C_engine.nn.batch_norm(_unwrap(x), w, b, eps))


def layer_norm(
    x: Tensor,
    normalized_shape: list[int] | tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Layer normalization."""
    w = _unwrap(weight) if weight is not None else None
    b = _unwrap(bias) if bias is not None else None
    # Engine API: layer_norm(x, gamma, beta, eps) — no normalized_shape arg
    return _wrap(_C_engine.nn.layer_norm(_unwrap(x), w, b, eps))


def group_norm(
    x: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Group normalization.  Engine signature: (x, gamma, beta, num_groups, eps)."""
    from lucid._factories.creation import ones, zeros
    C = x.shape[1]
    w = _unwrap(weight) if weight is not None else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    b = _unwrap(bias) if bias is not None else _unwrap(zeros(C, device=x.device, dtype=x.dtype))
    return _wrap(_C_engine.nn.group_norm(_unwrap(x), w, b, num_groups, eps))


def rms_norm(
    x: Tensor,
    normalized_shape: list[int] | tuple[int, ...],
    weight: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """RMS normalization.  Engine signature: (x, gamma, eps)."""
    from lucid._factories.creation import ones
    C = x.shape[-1]
    w = _unwrap(weight) if weight is not None else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    return _wrap(_C_engine.nn.rms_norm(_unwrap(x), w, eps))


def instance_norm(
    x: Tensor,
    running_mean: Tensor | None = None,
    running_var: Tensor | None = None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_input_stats: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Instance normalization (batch_norm with training=True and per-sample stats)."""
    return batch_norm(
        x,
        running_mean,
        running_var,
        weight,
        bias,
        training=True,
        momentum=momentum,
        eps=eps,
    )
