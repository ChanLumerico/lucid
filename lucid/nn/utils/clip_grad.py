"""
Gradient clipping utilities.
"""

from typing import Iterable, TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.parameter import Parameter


def clip_grad_norm_(
    parameters: Iterable[Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tensor:
    """Clip gradient norm of parameters in-place. Returns the total norm.

    Args:
        parameters:        Iterable of Parameters (grads clipped in-place).
        max_norm:          Maximum allowed gradient norm.
        norm_type:         Type of norm (2.0 = L2, float('inf') = max).
        error_if_nonfinite: Raise if total norm is nan/inf before clipping.

    Returns:
        Total gradient norm as a scalar Tensor.
    """
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _wrap

    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        import lucid
        return lucid.zeros(1)

    # Collect raw gradient numpy arrays (in-place-modifiable views into C++ buffer)
    raw_grads = [p._impl.grad_as_python() for p in params_with_grad]

    if norm_type == math.inf:
        total_norm = float(max(np.max(np.abs(g)) for g in raw_grads))
    else:
        total_norm = float(
            sum(float(np.sum(np.abs(g) ** norm_type)) for g in raw_grads)
            ** (1.0 / norm_type)
        )

    if error_if_nonfinite and (math.isnan(total_norm) or math.isinf(total_norm)):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients is "
            f"non-finite ({total_norm})."
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in raw_grads:
            g[:] *= clip_coef  # modify the gradient buffer in-place

    total_arr = np.array([total_norm], dtype=np.float32)
    return _wrap(_C_engine.TensorImpl(total_arr, _C_engine.Device.CPU, False))


def clip_grad_value_(
    parameters: Iterable[Parameter],
    clip_value: float,
) -> None:
    """Clip each gradient element to [-clip_value, clip_value] in-place.

    Args:
        parameters: Iterable of Parameters.
        clip_value: Absolute clipping bound.
    """
    for p in parameters:
        if p.grad is not None:
            g = p._impl.grad_as_python()
            np.clip(g, -clip_value, clip_value, out=g)
