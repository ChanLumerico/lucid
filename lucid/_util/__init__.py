from typing import Literal, Sequence, overload

from lucid._tensor import Tensor
from lucid.types import _ShapeLike, _ArrayLikeInt, _Scalar

from lucid._util import func


# fmt: off
__all__ = [
    "reshape", "squeeze", "unsqueeze", "expand_dims", "ravel", "stack", "hstack",
    "vstack", "concatenate", "pad", "repeat", "tile", "flatten", "meshgrid", 
    "split", "tril", "triu", "broadcast_to", "chunk", "masked_fill", "roll", 
    "unbind", "sort", "nonzero", "unique", "topk", "argsort", "histogramdd", 
    "histogram", "histogram2d", "where", "nonzero", "argmin", "argmax"
]
# fmt: on


def reshape(a: Tensor, /, shape: _ShapeLike) -> Tensor:
    return func.reshape(shape)(a)


@overload
def reshape_immediate(a: Tensor, /, shape: _ShapeLike) -> Tensor: ...


def _reshape_immediate(a: Tensor, /, *shape: int | _ShapeLike) -> Tensor:
    if isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return func._reshape_immediate(shape)(a)


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
    a: Tensor, /, size_or_sections: int | list[int] | tuple[int], axis: int = 0
) -> tuple[Tensor, ...]:
    return func.split(size_or_sections, axis)(a)


def tril(a: Tensor, /, diagonal: int = 0) -> Tensor:
    return func.tril(diagonal)(a)


def triu(a: Tensor, /, diagonal: int = 0) -> Tensor:
    return func.triu(diagonal)(a)


def broadcast_to(a: Tensor, /, shape: _ShapeLike) -> Tensor:
    return func.broadcast_to(shape)(a)


def chunk(a: Tensor, /, chunks: int, axis: int = 0) -> tuple[Tensor, ...]:
    return func.chunk(chunks, axis)(a)


def masked_fill(a: Tensor, /, mask: Tensor, value: _Scalar) -> Tensor:
    return func.masked_fill(mask, value)(a)


def roll(
    a: Tensor,
    /,
    shifts: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> Tensor:
    return func.roll(shifts, axis)(a)


def unbind(a: Tensor, /, axis: int = 0) -> tuple[Tensor, ...]:
    return func.unbind(axis)(a)


_SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]


def sort(
    a: Tensor,
    /,
    axis: int = -1,
    descending: bool = False,
    kind: _SortKind | None = None,
    stable: bool = False,
) -> tuple[Tensor, Tensor]:
    return func.sort(axis, descending, kind, stable)(a)


def argsort(
    a: Tensor,
    /,
    axis: int = -1,
    descending: bool = False,
    kind: _SortKind | None = None,
    stable: bool = False,
) -> Tensor:
    return func.argsort(axis, descending, kind, stable)(a)


def argmin(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    return func.argmin(axis, keepdims)(a)


def argmax(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    return func.argmax(axis, keepdims)(a)


def nonzero(a: Tensor, /) -> Tensor:
    return func.nonzero()(a)


def unique(
    a: Tensor,
    /,
    sorted: bool = True,
    axis: int | None = None,
    return_inverse: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    op = func.unique(sorted, axis)
    ret = op(a)
    if return_inverse:
        assert (
            op.inverse_ is not None
        ), "inverse_ was not computed; check axis/sorted constraints"
        return ret, op.inverse_
    else:
        return ret


def topk(
    a: Tensor, /, k: int, axis: int = -1, largest: bool = True, sorted: bool = True
) -> tuple[Tensor, Tensor]:
    if k > a.shape[axis]:
        raise ValueError(
            f"k={k} is greater than dimension size {a.shape[axis]} along axis {axis}."
        )
    return func.topk(k, axis, largest, sorted)(a)


def histogramdd(
    a: Tensor,
    /,
    bins: int | list[int],
    range: list[tuple[float, float]],
    density: bool = False,
) -> tuple[Tensor, Tensor]:
    if isinstance(bins, int):
        bins = [bins] * a.shape[1]
    return func.histogramdd(bins, range, density)(a)


def histogram(
    a: Tensor,
    /,
    bins: int = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
) -> tuple[Tensor, Tensor]:
    if a.ndim != 1:
        raise ValueError("histogram() expects a 1D tensor input.")
    a = a.reshape(-1, 1)
    range = [range or (float(a.data.min().item()), float(a.data.max().item()))]
    return func.histogramdd([bins], range, density)(a)


def histogram2d(
    a: Tensor,
    b: Tensor,
    /,
    bins: list[int, int] = [10, 10],
    range: list[tuple[float, float]] | None = None,
    density: bool = False,
) -> tuple[Tensor, Tensor]:
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    ab = stack((a, b), axis=1)
    if range is None:
        range = [
            (float(a.data.min().item()), float(a.data.max().item())),
            (float(b.data.min().item()), float(b.data.max().item())),
        ]
    return func.histogramdd(bins, range, density)(ab)


def where(condition: Tensor, a: Tensor, b: Tensor, /) -> Tensor:
    return func.where()(condition, a, b)


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
Tensor.unbind = unbind
Tensor.sort = sort
Tensor.unique = unique
Tensor.nonzero = nonzero
