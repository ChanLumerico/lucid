from typing import Any, Literal

import lucid
from lucid._tensor import Tensor
from lucid.nn.init import _dist
from lucid.types import _Scalar

_FanMode = Literal["fan_in", "fan_out"]


def _tensor_check(value: Any) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected value to be Tensor got {type(value).__name__}.")


def uniform_(tensor: Tensor, a: _Scalar = 0, b: _Scalar = 1) -> None:
    _tensor_check(tensor)
    return _dist.uniform_(tensor, a, b)


def normal_(tensor: Tensor, mean: _Scalar = 0.0, std: _Scalar = 1.0) -> None:
    _tensor_check(tensor)
    return _dist.normal_(tensor, mean, std)


def constant_(tensor: Tensor, val: _Scalar) -> None:
    _tensor_check(tensor)
    return _dist.constant_(tensor, val)


def xavier_uniform_(tensor: Tensor, gain: _Scalar = 1.0) -> None:
    _tensor_check(tensor)
    return _dist.xavier_uniform_(tensor, gain)


def xavier_normal_(tensor: Tensor, gain: _Scalar = 1.0) -> None:
    _tensor_check(tensor)
    return _dist.xavier_normal_(tensor, gain)


def kaiming_uniform_(tensor: Tensor, mode: _FanMode = "fan_in") -> None:
    _tensor_check(tensor)
    if mode not in {"fan_in", "fan_out"}:
        raise ValueError("mode must be either 'fan_in' or 'fan_out'.")

    return _dist.kaiming_uniform_(tensor, mode)


def kaiming_normal_(tensor: Tensor, mode: _FanMode = "fan_in") -> None:
    _tensor_check(tensor)
    if mode not in {"fan_in", "fan_out"}:
        raise ValueError("mode must be either 'fan_in' or 'fan_out'.")

    return _dist.kaiming_normal_(tensor, mode)
