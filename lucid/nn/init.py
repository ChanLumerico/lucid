"""
nn.init: parameter initialization functions.
All functions operate in-place and return the tensor.
"""

import math
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import _impl_with_grad as _iwg

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _fill_from_impl(tensor: Tensor, src_impl: object) -> Tensor:
    """Replace tensor's impl with src_impl, preserving requires_grad."""
    rg = tensor._impl.requires_grad
    impl = _C_engine.reshape(src_impl, list(tensor.shape))  # type: ignore[attr-defined]
    tensor._impl = _iwg(impl, rg)
    return tensor


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fill tensor in-place with values from U(a, b)."""
    return _fill_from_impl(
        tensor,
        _C_engine.uniform(
            list(tensor.shape), a, b, tensor._impl.dtype, tensor._impl.device
        ),
    )


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill tensor in-place with values from N(mean, std²)."""
    return _fill_from_impl(
        tensor,
        _C_engine.normal(
            list(tensor.shape), mean, std, tensor._impl.dtype, tensor._impl.device
        ),
    )


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Fill tensor in-place with a constant value."""
    return _fill_from_impl(
        tensor,
        _C_engine.full(
            list(tensor.shape), val, tensor._impl.dtype, tensor._impl.device
        ),
    )


def ones_(tensor: Tensor) -> Tensor:
    """Fill tensor in-place with ones."""
    return _fill_from_impl(
        tensor,
        _C_engine.ones(list(tensor.shape), tensor._impl.dtype, tensor._impl.device),
    )


def zeros_(tensor: Tensor) -> Tensor:
    """Fill tensor in-place with zeros."""
    return _fill_from_impl(
        tensor,
        _C_engine.zeros(list(tensor.shape), tensor._impl.dtype, tensor._impl.device),
    )


def eye_(tensor: Tensor) -> Tensor:
    """Fill a 2D tensor in-place as an identity matrix."""
    if tensor.ndim != 2:
        raise ValueError("eye_() requires a 2D tensor")
    n, m = tensor.shape
    return _fill_from_impl(
        tensor, _C_engine.eye(n, m, 0, tensor._impl.dtype, tensor._impl.device)
    )


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with Xavier uniform values."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with Xavier normal values."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    """Fill tensor with Kaiming uniform values."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    """Fill tensor with Kaiming normal values."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> Tensor:
    """Fill tensor with truncated normal values via rejection sampling."""
    shape = list(tensor.shape) if tensor.shape else [1]
    total = 1
    for s in shape:
        total *= s

    dt = tensor._impl.dtype
    dev = tensor._impl.device
    filled_parts = []
    remaining = total
    while remaining > 0:
        needed = max(remaining * 4, 16)
        candidates = _C_engine.normal([needed], mean, std, dt, dev)
        # Build mask: a <= x <= b
        lo = _C_engine.full([needed], a, dt, dev)
        hi = _C_engine.full([needed], b, dt, dev)
        ge_a = _C_engine.greater_equal(candidates, lo)
        le_b = _C_engine.less_equal(candidates, hi)
        mask = _C_engine.bitwise_and(ge_a, le_b)
        valid = _C_engine.masked_select(candidates, mask)
        n_valid = int(list(valid.shape)[0])
        take = min(n_valid, remaining)
        if take > 0:
            # Slice first `take` elements via gather + arange.
            idx = _C_engine.arange(0, take, 1, _C_engine.I32, dev)
            filled_parts.append(_C_engine.gather(valid, idx, 0))
            remaining -= take

    result = _C_engine.concatenate(filled_parts, 0)
    return _fill_from_impl(tensor, result)


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with a (semi-)orthogonal matrix via SVD."""
    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    dt = tensor._impl.dtype
    dev = tensor._impl.device
    flat = _C_engine.normal([rows, cols], 0.0, 1.0, dt, dev)
    # linalg.svd returns (U, S, Vt) as a Python tuple.
    svd_result = _C_engine.linalg.svd(flat, True)
    U = svd_result[0]  # (rows, min(rows,cols))
    Vt = svd_result[2]  # (min(rows,cols), cols)
    q = U if rows < cols else Vt
    # Slice q to (rows, cols) via gather on both axes.
    r_idx = _C_engine.arange(0, rows, 1, _C_engine.I32, dev)
    c_idx = _C_engine.arange(0, cols, 1, _C_engine.I32, dev)
    q = _C_engine.gather(q, r_idx, 0)
    q = _C_engine.gather(q, c_idx, 1)
    if gain != 1.0:
        g = _C_engine.full([rows, cols], gain, dt, dev)
        q = _C_engine.mul(q, g)
    return _fill_from_impl(tensor, q)


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Return the recommended gain for an activation function."""
    _gains: dict[str, float] = {
        "linear": 1.0,
        "conv1d": 1.0,
        "conv2d": 1.0,
        "conv3d": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": math.sqrt(2.0),
        "selu": 3.0 / 4.0,
    }
    if nonlinearity == "leaky_relu":
        slope = param if param is not None else 0.01
        return math.sqrt(2.0 / (1 + slope**2))
    if nonlinearity in _gains:
        return _gains[nonlinearity]
    raise ValueError(f"Unsupported nonlinearity: {nonlinearity!r}")


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Compute fan_in and fan_out for Linear and Conv weight tensors."""
    ndim = tensor.ndim
    if ndim < 2:
        raise ValueError("fan_in/fan_out requires at least 2D tensor")
    receptive = 1
    if ndim > 2:
        for s in tensor.shape[2:]:
            receptive *= s
    fan_in = tensor.shape[1] * receptive
    fan_out = tensor.shape[0] * receptive
    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        return fan_in
    if mode == "fan_out":
        return fan_out
    raise ValueError(f"Unknown mode: {mode!r}")
