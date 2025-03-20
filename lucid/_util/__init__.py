from typing import Literal, Sequence
from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLikeInt, _Scalar

from lucid._util import func


# fmt: off
__all__ = [
    "reshape", "squeeze", "unsqueeze", "expand_dims", "ravel", "stack", "hstack",
    "vstack", "concatenate", "pad", "repeat", "tile", "flatten", "meshgrid"
]
# fmt: on


def reshape(a: Tensor, /, shape: _ShapeLike) -> Tensor:
    return func.reshape(shape)(a)


def _reshape_immediate(a: Tensor, /, *shape: int) -> Tensor:
    return func._reshape_immediate(*shape)(a)


def squeeze(a: Tensor, /, axis: _ShapeLike | None = None) -> Tensor:
    return func.squeeze(axis)(a)


def unsqueeze(a: Tensor, /, axis: _ShapeLike) -> Tensor:
    return func.unsqueeze(axis)(a)


def expand_dims(a: Tensor, /, axis: _ShapeLike) -> Tensor:
    return func.expand_dims(axis)(a)


def ravel(a: Tensor, /) -> Tensor:
    return func.ravel()(a)


def stack(arr: tuple[Tensor, ...], /, axis: int = 0) -> Tensor:
    return func.stack(axis)(*arr)


def hstack(arr: tuple[Tensor, ...], /) -> Tensor:
    return func.hstack()(*arr)


def vstack(arr: tuple[Tensor, ...], /) -> Tensor:
    return func.vstack()(*arr)


def concatenate(arr: tuple[Tensor, ...], /, axis: int = 0) -> Tensor:
    return func.concatenate(axis)(*arr)


def pad(a: Tensor, /, pad_width: _ArrayLikeInt) -> Tensor:
    return func.pad(pad_width, ndim=a.ndim)(a)


def repeat(
    a: Tensor, /, repeats: int | Sequence[int], axis: int | None = None
) -> Tensor:
    return func.repeat(repeats, axis)(a)


def tile(a: Tensor, /, reps: int | Sequence[int]) -> Tensor:
    return func.tile(reps)(a)


def flatten(a: Tensor, /, axis: int = 0) -> Tensor:
    return func.flatten(axis)(a)


def meshgrid(
    a: Tensor, b: Tensor, /, indexing: Literal["xy", "ij"] = "ij"
) -> tuple[Tensor, Tensor]:
    return func.meshgrid(indexing)(a, b)


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


Tensor.reshape = _reshape_immediate
Tensor.squeeze = squeeze
Tensor.unsqueeze = unsqueeze
Tensor.ravel = ravel
Tensor.pad = pad
Tensor.repeat = repeat
Tensor.tile = tile
Tensor.flatten = flatten
Tensor.split = split
Tensor.tril = tril
Tensor.triu = triu
Tensor.broadcast_to = broadcast_to
Tensor.chunk = chunk
Tensor.masked_fill = masked_fill
Tensor.roll = roll
