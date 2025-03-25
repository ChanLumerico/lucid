from typing import Literal

from lucid._tensor import Tensor
from lucid.einops import _func

from lucid.types import _EinopsPattern


__all__ = ["rearrange", "reduce", "repeat"]


_ReduceStr = Literal["sum", "mean"]


def rearrange(a: Tensor, /, pattern: _EinopsPattern, **shapes: int) -> Tensor:
    return _func.rearrange(pattern, t_shape=a.shape, **shapes)(a)


def reduce(
    a: Tensor, /, pattern: _EinopsPattern, reduction: _ReduceStr = "sum", **shapes: int
) -> Tensor:
    return _func.reduce(pattern, reduction, t_shape=a.shape, **shapes)(a)


def repeat(a: Tensor, /, pattern: _EinopsPattern, **shapes: int) -> Tensor:
    return _func.repeat(pattern, t_shape=a.shape, **shapes)(a)
