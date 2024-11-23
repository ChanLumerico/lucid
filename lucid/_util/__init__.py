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


def stack(arr: tuple[Tensor, ...], axis: int = 0) -> Tensor:
    return func.stack(*arr, axis=axis)


def hstack(arr: tuple[Tensor, ...]) -> Tensor:
    return func.hstack(*arr)


def vstack(arr: tuple[Tensor, ...]) -> Tensor:
    return func.vstack(*arr)


def concatenate(arr: tuple[Tensor, ...], axis: int = 0) -> Tensor:
    return func.concatenate(*arr, axis=axis)


Tensor.reshape = func._reshape_inplace
Tensor.squeeze = func.squeeze
Tensor.unsqueeze = func.unsqueeze
Tensor.ravel = func.ravel
