import lucid

from lucid._tensor import Tensor
from lucid.nn.init import _dist
from lucid.types import _Scalar


def uniform_(tensor: Tensor, a: _Scalar, b: _Scalar) -> None:
    return _dist.uniform_(tensor, a, b)
