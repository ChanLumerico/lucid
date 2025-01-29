from typing import Self, Sequence

from lucid.types import _Scalar, _ArrayOrScalar, _ShapeLike, _ArrayLikeInt


class _TensorOps:
    def __add__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __radd__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __sub__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __rsub__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __mul__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __rmul__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __truediv__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __rtruediv__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __matmul__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __eq__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __ne__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __gt__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __ge__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __lt__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __le__(self, other: Self | _ArrayOrScalar) -> Self: ...

    def __pow__(self, _: _Scalar) -> Self: ...

    def __neg__(self) -> Self: ...

    @property
    def T(self) -> Self: ...

    @property
    def mT(self) -> Self: ...

    def dot(self, other: Self | _ArrayOrScalar) -> Self: ...

    def matmul(self, other: Self) -> Self: ...

    def sum(
        self, axis: int | tuple[int] | None = None, keepdims: bool = False
    ) -> Self: ...

    def mean(
        self, axis: int | tuple[int] | None = None, keepdims: bool = False
    ) -> Self: ...

    def var(
        self, axis: int | tuple[int] | None = None, keepdims: bool = False
    ) -> Self: ...

    def clip(self, min_value: _Scalar, max_value: _Scalar) -> Self: ...

    def reshape(self, *shape: int) -> Self: ...

    def transpose(self, axes: Sequence[int]) -> Self: ...

    def squeeze(self, axis: _ShapeLike | None = None) -> Self: ...

    def unsqueeze(self, axis: _ShapeLike) -> Self: ...

    def ravel(self) -> Self: ...

    def pad(self, pad_width: _ArrayLikeInt) -> Self: ...

    def repeat(self, repeats: int | Sequence[int], axis: int | None = None) -> Self: ...

    def tile(self, reps: int | Sequence[int]) -> Self: ...

    def flatten(self: Self) -> Self: ...

    def split(
        self: Self, size_or_sections: int | Sequence[int], axis: int = 0
    ) -> tuple[Self, ...]: ...
