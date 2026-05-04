"""
Gradient clipping utilities.
All computation goes through the C++ engine — no numpy.
"""

from typing import Iterable, TYPE_CHECKING
import math
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._factories.creation import zeros

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.parameter import Parameter


def clip_grad_norm_(
    parameters: Iterable["Parameter"],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> "Tensor":
    """Clip gradient norm of parameters in-place. Returns the total norm."""
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return zeros(1)

    dev = params_with_grad[0]._impl.device
    dt = params_with_grad[0]._impl.dtype

    if norm_type == math.inf:
        max_val = _C_engine.full([1], float("-inf"), dt, dev)
        for p in params_with_grad:
            g_impl = _unwrap(p.grad)
            abs_g = _C_engine.abs(g_impl)
            m = _C_engine.reshape(_C_engine.max(abs_g, [], False), [1])
            mv = _C_engine.reshape(max_val, [1])
            max_val = _C_engine.max(_C_engine.stack([mv, m], 0), [0], False)
        total_norm = float(_wrap(max_val).item())
    else:
        acc = _C_engine.zeros([1], dt, dev)
        for p in params_with_grad:
            g_impl = _unwrap(p.grad)
            pow_g = _C_engine.pow_scalar(_C_engine.abs(g_impl), norm_type)
            s = _C_engine.reshape(_C_engine.sum(pow_g, [], False), [1])
            acc = _C_engine.add(acc, s)
        total_norm = float(_wrap(acc).item()) ** (1.0 / norm_type)

    if error_if_nonfinite and (math.isnan(total_norm) or math.isinf(total_norm)):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients is "
            f"non-finite ({total_norm})."
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grad:
            g_impl = _unwrap(p.grad)
            coef = _C_engine.full(
                list(g_impl.shape), clip_coef, g_impl.dtype, g_impl.device
            )
            p._impl.set_grad(_C_engine.mul(g_impl, coef))

    return _wrap(_C_engine.full([1], total_norm, dt, dev))


def clip_grad_value_(
    parameters: Iterable["Parameter"],
    clip_value: float,
) -> None:
    """Clip each gradient element to [-clip_value, clip_value] in-place."""
    for p in parameters:
        if p.grad is not None:
            g_impl = _unwrap(p.grad)
            p._impl.set_grad(_C_engine.clip(g_impl, -clip_value, clip_value))
