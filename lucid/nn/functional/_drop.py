"""
lucid.nn.functional._drop — dropout family.

All routes are 1:1 to C++ engine ops — no Python compositions.
"""

from __future__ import annotations

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of


def _prob_check(p: float) -> None:
    if not 0 <= p < 1:
        raise ValueError("Dropout probability `p` must be in the range [0, 1).")


def dropout(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    _prob_check(p)
    return Tensor._wrap(_C_nn.dropout(impl_of(input_), float(p), bool(training), None))


def dropoutnd(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    _prob_check(p)
    return Tensor._wrap(_C_nn.dropoutnd(impl_of(input_), float(p), bool(training), None))


def alpha_dropout(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    _prob_check(p)
    return Tensor._wrap(
        _C_nn.alpha_dropout(impl_of(input_), float(p), bool(training), None))


def drop_block(
    input_: Tensor, block_size: int, p: float = 0.1, eps: float = 1e-7
) -> Tensor:
    _prob_check(p)
    if input_.ndim != 4:
        raise ValueError("drop_block: input must be 4-D (N, C, H, W).")
    return Tensor._wrap(_C_nn.drop_block(
        impl_of(input_), int(block_size), float(p), float(eps), None))


def drop_path(input_: Tensor, p: float = 0.1, scale_by_keep: bool = True) -> Tensor:
    _prob_check(p)
    return Tensor._wrap(_C_nn.drop_path(
        impl_of(input_), float(p), bool(scale_by_keep), None))
