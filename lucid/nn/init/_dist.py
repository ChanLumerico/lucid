from typing import Any

import lucid

from lucid._tensor import Tensor
from lucid.types import _Scalar


def _tensor_check(value: Any) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"Expected value to be Tensor got {type(value).__name__}.")


def uniform_(tensor: Tensor, a: _Scalar, b: _Scalar) -> None:
    _tensor_check(tensor)
    ...
