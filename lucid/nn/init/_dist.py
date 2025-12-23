from functools import reduce
import numpy as np

from lucid._backend.metal import mx
from lucid._tensor import Tensor
from lucid.types import _Scalar


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
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
            + "Must be at least 2D."
        )

    return fan_in, fan_out


def _assign_like(tensor: Tensor, data: np.ndarray) -> None:
    if tensor.is_cpu():
        tensor.data = np.asarray(data, dtype=tensor.data.dtype)
    else:
        tensor.data = mx.array(data, dtype=tensor.data.dtype)


def uniform(tensor: Tensor, a: _Scalar, b: _Scalar) -> None:
    data = np.random.uniform(a, b, size=tensor.shape)
    _assign_like(tensor, data)


def normal(tensor: Tensor, mean: _Scalar, std: _Scalar) -> None:
    data = np.random.normal(mean, std, size=tensor.shape)
    _assign_like(tensor, data)


def constant(tensor: Tensor, val: _Scalar) -> None:
    data = np.full(tensor.shape, val)
    _assign_like(tensor, data)


def xavier_uniform(tensor: Tensor, gain: _Scalar) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = (6 / (fan_in + fan_out)) ** 0.5 * gain
    data = np.random.uniform(-bound, bound, size=tensor.shape)
    _assign_like(tensor, data)


def xavier_normal(tensor: Tensor, gain: _Scalar) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = (2 / (fan_in + fan_out)) ** 0.5 * gain
    data = np.random.normal(0.0, std, size=tensor.shape)
    _assign_like(tensor, data)


def kaiming_uniform(tensor: Tensor, mode: str) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    bound = (6 / fan) ** 0.5
    data = np.random.uniform(-bound, bound, size=tensor.shape)
    _assign_like(tensor, data)


def kaiming_normal(tensor: Tensor, mode: str) -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    std = (2 / fan) ** 0.5
    data = np.random.normal(0.0, std, size=tensor.shape)
    _assign_like(tensor, data)
