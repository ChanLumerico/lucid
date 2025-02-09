from typing import Literal

from lucid._tensor import Tensor
from lucid.einops import _func

from lucid.types import _EinopsPattern

_ReduceStr = Literal["sum", "mean"]


def rearrange(a: Tensor, pattern: _EinopsPattern, **shapes: int) -> Tensor:
    return _func.rearrange(a, pattern, **shapes)


def reduce(
    a: Tensor, pattern: _EinopsPattern, reduction: _ReduceStr = "sum", **shapes: int
) -> Tensor:
    return _func.reduce(a, pattern, reduction, **shapes)


def repeat(a: Tensor, pattern: _EinopsPattern, **shapes: int) -> Tensor:
    return _func.repeat(a, pattern, **shapes)
