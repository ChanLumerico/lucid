from lucid._tensor import Tensor
from lucid.types import _ShapeLike

from lucid._util import func


def reshape(a: Tensor, shape: _ShapeLike) -> Tensor:
    return func._reshape(a, shape)


def squeeze(a: Tensor, axis: _ShapeLike | None = None) -> Tensor:
    return func.squeeze(a, axis)


def unsqueeze(a: Tensor, axis: _ShapeLike) -> Tensor:
    return func.unsqueeze(a, axis)


def ravel(a: Tensor) -> Tensor:
    return func.ravel(a)


Tensor.reshape = func._reshape_inplace
Tensor.squeeze = func.squeeze
Tensor.unsqueeze = func.unsqueeze
Tensor.ravel = func.ravel
