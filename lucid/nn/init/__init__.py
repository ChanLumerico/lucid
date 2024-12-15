from typing import Any, Literal

import lucid
from lucid._tensor import Tensor
from lucid.nn.init import _dist
from lucid.types import _Scalar

_FanMode = Literal["fan_in", "fan_out"]


def _tensor_check(value: Any) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected value to be Tensor got {type(value).__name__}.")


def uniform(tensor: Tensor, a: _Scalar = 0, b: _Scalar = 1) -> None:
    _tensor_check(tensor)
    return _dist.uniform(tensor, a, b)


def normal(tensor: Tensor, mean: _Scalar = 0.0, std: _Scalar = 1.0) -> None:
    _tensor_check(tensor)
    return _dist.normal(tensor, mean, std)


def constant(tensor: Tensor, val: _Scalar) -> None:
    _tensor_check(tensor)
    return _dist.constant(tensor, val)


def xavier_uniform(tensor: Tensor, gain: _Scalar = 1.0) -> None:
    _tensor_check(tensor)
    return _dist.xavier_uniform(tensor, gain)


def xavier_normal(tensor: Tensor, gain: _Scalar = 1.0) -> None:
    _tensor_check(tensor)
    return _dist.xavier_normal(tensor, gain)


def kaiming_uniform(tensor: Tensor, mode: _FanMode = "fan_in") -> None:
    _tensor_check(tensor)
    if mode not in {"fan_in", "fan_out"}:
        raise ValueError("mode must be either 'fan_in' or 'fan_out'.")

    return _dist.kaiming_uniform(tensor, mode)


def kaiming_normal(tensor: Tensor, mode: _FanMode = "fan_in") -> None:
    _tensor_check(tensor)
    if mode not in {"fan_in", "fan_out"}:
        raise ValueError("mode must be either 'fan_in' or 'fan_out'.")

    return _dist.kaiming_normal(tensor, mode)
