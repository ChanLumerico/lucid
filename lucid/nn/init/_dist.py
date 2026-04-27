"""
lucid.nn.init._dist — distribution backends.

Each function fills the tensor in-place from numpy-side draws, then
swaps the new buffer into the tensor's `_impl`. This preserves
identity (same Parameter/Buffer object), so any external references
(optimizer state, etc.) remain valid.
"""

from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np

from lucid._tensor import Tensor
from lucid.types import _Scalar


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Mirrors PyTorch / legacy fan calculation for 2D / conv weights."""
    if tensor.ndim == 2:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    elif tensor.ndim in {3, 4, 5}:
        kernel_prod = reduce(lambda x, y: x * y, tensor.shape[2:], 1)
        fan_in = tensor.shape[1] * kernel_prod
        fan_out = tensor.shape[0] * kernel_prod
    else:
        raise ValueError(
            f"Tensor with dims {tensor.ndim} is not supported. "
            "Must be at least 2D."
        )
    return fan_in, fan_out


def _assign_like(tensor: Tensor, data: np.ndarray) -> None:
    """Build a fresh Tensor from `data` and swap it into `tensor._impl`.

    Keeps Parameter identity so optimizers / module registries stay valid.
    """
    new = Tensor(data, dtype=tensor.dtype, device=tensor.device)
    tensor._impl = new._impl


def uniform(tensor: Tensor, a: _Scalar, b: _Scalar) -> None:
    _assign_like(tensor, np.random.uniform(a, b, size=tensor.shape))


def normal(tensor: Tensor, mean: _Scalar, std: _Scalar) -> None:
    _assign_like(tensor, np.random.normal(mean, std, size=tensor.shape))


def constant(tensor: Tensor, val: _Scalar) -> None:
    _assign_like(tensor, np.full(tensor.shape, val, dtype=np.float32))


def xavier_uniform(tensor: Tensor, gain: _Scalar) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = (6 / (fan_in + fan_out)) ** 0.5 * gain
    _assign_like(tensor, np.random.uniform(-bound, bound, size=tensor.shape))


def xavier_normal(tensor: Tensor, gain: _Scalar) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = (2 / (fan_in + fan_out)) ** 0.5 * gain
    _assign_like(tensor, np.random.normal(0.0, std, size=tensor.shape))


def kaiming_uniform(tensor: Tensor, mode: str) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    bound = (6 / fan) ** 0.5
    _assign_like(tensor, np.random.uniform(-bound, bound, size=tensor.shape))


def kaiming_normal(tensor: Tensor, mode: str) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    std = (2 / fan) ** 0.5
    _assign_like(tensor, np.random.normal(0.0, std, size=tensor.shape))
