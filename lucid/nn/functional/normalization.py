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

    Dispatches to the correct engine op by input dimensionality:
      ndim=2  (N, C)         → batch_norm1d  [unsqueeze dim-2, squeeze after]
      ndim=3  (N, C, L)      → batch_norm1d
      ndim=4  (N, C, H, W)   → batch_norm
      ndim=5  (N, C, D, H, W)→ batch_norm3d

    Eval mode (training=False) uses running_mean/running_var when available.
    """
    from lucid._factories.creation import ones, zeros

    C = x.shape[1]
    ndim = x.ndim
    xi = _unwrap(x)
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    )
    b = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(C, device=x.device, dtype=x.dtype))
    )

    # ── Eval mode: use precomputed running statistics ─────────────────────────
    if not training and running_mean is not None and running_var is not None:
        rm = _unwrap(running_mean)
        rv = _unwrap(running_var)
        out_impl = _C_engine.nn.batch_norm_eval(xi, rm, rv, w, b, eps)
        return _wrap(out_impl)

    # ── Training mode: dispatch by dimensionality ─────────────────────────────
    if ndim == 2:
        # (N, C) → unsqueeze to (N, C, 1), batch_norm1d, squeeze back
        xi_3d = _C_engine.unsqueeze(xi, 2)
        out_3d = _C_engine.nn.batch_norm1d(xi_3d, w, b, eps)
        return _wrap(_C_engine.squeeze(out_3d, 2))
    elif ndim == 3:
        return _wrap(_C_engine.nn.batch_norm1d(xi, w, b, eps))
    elif ndim == 4:
        return _wrap(_C_engine.nn.batch_norm(xi, w, b, eps))
    elif ndim == 5:
        return _wrap(_C_engine.nn.batch_norm3d(xi, w, b, eps))
    else:
        raise ValueError(f"batch_norm: expected 2–5D input, got ndim={ndim}")


def layer_norm(
    x: Tensor,
    normalized_shape: list[int] | tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Layer normalization.

    The engine op requires non-null gamma/beta; we materialise identity
    defaults (ones / zeros) when the user passes ``None`` so that
    ``LayerNorm(elementwise_affine=False)`` and ``LayerNorm(bias=False)``
    work transparently.
    """
    from lucid._factories.creation import ones, zeros

    shape: tuple[int, ...] = tuple(normalized_shape)
    w: object = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(*shape, device=x.device, dtype=x.dtype))
    )
    b: object = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(*shape, device=x.device, dtype=x.dtype))
    )
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
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    )
    b = (
        _unwrap(bias)
        if bias is not None
        else _unwrap(zeros(C, device=x.device, dtype=x.dtype))
    )
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
    w = (
        _unwrap(weight)
        if weight is not None
        else _unwrap(ones(C, device=x.device, dtype=x.dtype))
    )
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
    """Instance normalization.

    Reduces over the *spatial* dimensions only — every ``(n, c)`` slice is
    standardised against its own mean and variance.  Differs from
    ``batch_norm`` which also reduces over the batch axis.

    When ``use_input_stats=False`` and ``running_mean`` / ``running_var``
    are supplied, those running statistics replace per-instance stats —
    the same fallback the reference framework uses in eval mode with
    ``track_running_stats=True``.
    """
    import lucid as _lucid

    if x.ndim < 3:
        raise ValueError(
            f"instance_norm: expected at least 3-D input (N, C, *spatial), "
            f"got ndim={x.ndim}"
        )
    spatial_dims: list[int] = list(range(2, x.ndim))
    C: int = int(x.shape[1])
    # Channel-broadcast shape (1, C, 1, 1, ...) for affine + running stats.
    bcast_shape: list[int] = [1, C] + [1] * (x.ndim - 2)

    if use_input_stats or running_mean is None or running_var is None:
        # Per-instance (per-(n,c)) statistics.
        mean: Tensor = x.mean(spatial_dims, keepdim=True)
        var: Tensor = x.var(spatial_dims, keepdim=True, correction=0)
    else:
        mean = running_mean.reshape(bcast_shape)
        var = running_var.reshape(bcast_shape)

    y: Tensor = (x - mean) / (var + eps).sqrt()

    if weight is not None:
        y = y * weight.reshape(bcast_shape)
    if bias is not None:
        y = y + bias.reshape(bcast_shape)
    return y
