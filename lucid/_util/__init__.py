from typing import Literal, Sequence
from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLikeInt

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


def pad(a: Tensor, pad_width: _ArrayLikeInt) -> Tensor:
    return func.pad(a, pad_width)


def repeat(a: Tensor, repeats: int | Sequence[int], axis: int | None = None) -> Tensor:
    return func.repeat(a, repeats, axis=axis)


def tile(a: Tensor, reps: int | Sequence[int]) -> Tensor:
    return func.tile(a, reps)


def flatten(a: Tensor) -> Tensor:
    return func.flatten(a)


def meshgrid(
    self: Tensor, other: Tensor, indexing: Literal["xy", "ij"] = "xy"
) -> tuple[Tensor, Tensor]:
    return func.meshgrid(self, other, indexing)


Tensor.reshape = func._reshape_inplace
Tensor.squeeze = func.squeeze
Tensor.unsqueeze = func.unsqueeze
Tensor.ravel = func.ravel
Tensor.pad = func.pad
Tensor.repeat = func.repeat
Tensor.tile = func.tile
Tensor.flatten = func.flatten
