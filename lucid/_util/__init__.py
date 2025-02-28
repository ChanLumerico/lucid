from typing import Literal, Sequence
from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLikeInt, _Scalar

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
    a: Tensor, b: Tensor, indexing: Literal["xy", "ij"] = "ij"
) -> tuple[Tensor, Tensor]:
    return func.meshgrid(a, b, indexing)


def split(
    a: Tensor, size_or_sections: int | list[int] | tuple[int], axis: int = 0
) -> tuple[Tensor, ...]:
    return func.split(a, size_or_sections, axis)


def tril(a: Tensor, diagonal: int = 0) -> Tensor:
    return func.tril(a, diagonal)


def triu(a: Tensor, diagonal: int = 0) -> Tensor:
    return func.triu(a, diagonal)


def broadcast_to(a: Tensor, shape: _ShapeLike) -> Tensor:
    return func.broadcast_to(a, shape)


def chunk(input_: Tensor, chunks: int, axis: int = 0) -> tuple[Tensor, ...]:
    return func.chunk(input_, chunks, axis)


def masked_fill(input_: Tensor, mask: Tensor, value: _Scalar) -> Tensor:
    return func.masked_fill(input_, mask, value)


def roll(
    input_: Tensor,
    shifts: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> Tensor:
    return func.roll(input_, shifts, axis)


Tensor.reshape = func._reshape_inplace
Tensor.squeeze = func.squeeze
Tensor.unsqueeze = func.unsqueeze
Tensor.ravel = func.ravel
Tensor.pad = func.pad
Tensor.repeat = func.repeat
Tensor.tile = func.tile
Tensor.flatten = func.flatten
Tensor.split = func.split
Tensor.tril = func.tril
Tensor.triu = func.triu
Tensor.broadcast_to = func.broadcast_to
Tensor.chunk = func.chunk
Tensor.masked_fill = func.masked_fill
Tensor.roll = func.roll
