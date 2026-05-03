"""
nn.init: parameter initialization functions.
All functions operate in-place and return the tensor.
"""

import math
from typing import TYPE_CHECKING

import numpy as np

from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _fill_impl(tensor: "Tensor", arr: np.ndarray) -> "Tensor":  # type: ignore[type-arg]
    impl = _C_engine.TensorImpl(
        np.ascontiguousarray(arr.reshape(tensor.shape)),
        tensor._impl.device,
        tensor._impl.requires_grad,
    )
    tensor._impl = impl
    return tensor


def uniform_(tensor: "Tensor", a: float = 0.0, b: float = 1.0) -> "Tensor":
    """Fill tensor in-place with values from U(a, b)."""
    arr = np.random.uniform(a, b, size=tensor.shape).astype("float32")
    return _fill_impl(tensor, arr)


def normal_(tensor: "Tensor", mean: float = 0.0, std: float = 1.0) -> "Tensor":
    """Fill tensor in-place with values from N(mean, std²)."""
    arr = np.random.normal(mean, std, size=tensor.shape).astype("float32")
    return _fill_impl(tensor, arr)


def constant_(tensor: "Tensor", val: float) -> "Tensor":
    """Fill tensor in-place with a constant value."""
    arr = np.full(tensor.shape, val, dtype="float32")
    return _fill_impl(tensor, arr)


def ones_(tensor: "Tensor") -> "Tensor":
    """Fill tensor in-place with ones."""
    return constant_(tensor, 1.0)


def zeros_(tensor: "Tensor") -> "Tensor":
    """Fill tensor in-place with zeros."""
    return constant_(tensor, 0.0)


def eye_(tensor: "Tensor") -> "Tensor":
    """Fill a 2D tensor in-place as an identity matrix."""
    if tensor.ndim != 2:
        raise ValueError("eye_() requires a 2D tensor")
    n, m = tensor.shape
    arr = np.eye(n, m, dtype="float32")
    return _fill_impl(tensor, arr)


def xavier_uniform_(tensor: "Tensor", gain: float = 1.0) -> "Tensor":
    """Fill tensor with Xavier uniform values."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor: "Tensor", gain: float = 1.0) -> "Tensor":
    """Fill tensor with Xavier normal values."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor: "Tensor",
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> "Tensor":
    """Fill tensor with Kaiming uniform values."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: "Tensor",
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> "Tensor":
    """Fill tensor with Kaiming normal values."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


def trunc_normal_(
    tensor: "Tensor",
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> "Tensor":
    """Fill tensor with truncated normal values."""
    from scipy.stats import truncnorm  # type: ignore[import-untyped]
    lo, hi = (a - mean) / std, (b - mean) / std
    arr = truncnorm.rvs(lo, hi, loc=mean, scale=std, size=tensor.shape).astype("float32")
    return _fill_impl(tensor, arr)


def orthogonal_(tensor: "Tensor", gain: float = 1.0) -> "Tensor":
    """Fill tensor with a (semi-)orthogonal matrix."""
    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    flat = np.random.normal(0, 1, (rows, cols)).astype("float32")
    u, _, v = np.linalg.svd(flat, full_matrices=False)
    q = u if rows < cols else v
    q = q[:rows, :cols].reshape(tensor.shape)
    return _fill_impl(tensor, q * gain)


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Return the recommended gain for an activation function."""
    _gains: dict[str, float] = {
        "linear":  1.0,
        "conv1d":  1.0,
        "conv2d":  1.0,
        "conv3d":  1.0,
        "sigmoid": 1.0,
        "tanh":    5.0 / 3.0,
        "relu":    math.sqrt(2.0),
        "selu":    3.0 / 4.0,
    }
    if nonlinearity == "leaky_relu":
        slope = param if param is not None else 0.01
        return math.sqrt(2.0 / (1 + slope ** 2))
    if nonlinearity in _gains:
        return _gains[nonlinearity]
    raise ValueError(f"Unsupported nonlinearity: {nonlinearity!r}")


def _calculate_fan_in_and_fan_out(tensor: "Tensor") -> tuple[int, int]:
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


def _calculate_correct_fan(tensor: "Tensor", mode: str) -> int:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        return fan_in
    if mode == "fan_out":
        return fan_out
    raise ValueError(f"Unknown mode: {mode!r}")
