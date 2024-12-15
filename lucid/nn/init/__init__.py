from typing import Any

import lucid
from lucid._tensor import Tensor
from lucid.nn.init import _dist
from lucid.types import _Scalar


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
