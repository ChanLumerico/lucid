"""
lucid.nn.functional._loss — fused loss kernels.

All routes are 1:1 to engine ops. Optional weight / pos_weight tensors
are filled with ones if not provided (the C++ ops require fixed-arity
inputs; the engine treats ones-weighted as the unweighted baseline).

Reduction encoding (matches engine `Reduction` enum):
    None → 0,  "mean" → 1,  "sum" → 2.
"""

from __future__ import annotations

from typing import Literal

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of
from lucid.ops.gfunc import ones
from lucid.types import Int64


_ReductionType = Literal["mean", "sum"]


def _reduction_code(reduction: _ReductionType | None) -> int:
    if reduction is None:
        return 0
    if reduction == "mean":
        return 1
    if reduction == "sum":
        return 2
    raise ValueError(f"Invalid reduction: {reduction!r}. Choose 'mean', 'sum', or None.")


def mse_loss(
    input_: Tensor, target: Tensor, reduction: _ReductionType | None = "mean"
) -> Tensor:
    return Tensor._wrap(_C_nn.mse_loss(
        impl_of(input_), impl_of(target), _reduction_code(reduction)))


def binary_cross_entropy(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    eps: float = 1e-7,
) -> Tensor:
    if weight is None:
        weight = ones(input_.shape, dtype=input_.dtype, device=input_.device)
    return Tensor._wrap(_C_nn.bce_loss(
        impl_of(input_), impl_of(target), impl_of(weight),
        _reduction_code(reduction), float(eps)))


def binary_cross_entropy_with_logits(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
) -> Tensor:
    if weight is None:
        weight = ones(input_.shape, dtype=input_.dtype, device=input_.device)
    if pos_weight is None:
        pos_weight = ones(input_.shape, dtype=input_.dtype, device=input_.device)
    return Tensor._wrap(_C_nn.bce_with_logits(
        impl_of(input_), impl_of(target), impl_of(weight), impl_of(pos_weight),
        _reduction_code(reduction)))


def cross_entropy(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    eps: float = 1e-7,
    ignore_index: int | None = None,
) -> Tensor:
    weight_impl = impl_of(weight) if weight is not None else None
    ig = -100 if ignore_index is None else int(ignore_index)
    return Tensor._wrap(_C_nn.cross_entropy_loss(
        impl_of(input_), impl_of(target), weight_impl,
        _reduction_code(reduction), float(eps), ig))


def nll_loss(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    ignore_index: int | None = None,
) -> Tensor:
    weight_impl = impl_of(weight) if weight is not None else None
    ig = -100 if ignore_index is None else int(ignore_index)
    return Tensor._wrap(_C_nn.nll_loss(
        impl_of(input_), impl_of(target), weight_impl,
        _reduction_code(reduction), ig))


def huber_loss(
    input_: Tensor,
    target: Tensor,
    delta: float = 1.0,
    reduction: _ReductionType | None = "mean",
) -> Tensor:
    return Tensor._wrap(_C_nn.huber_loss(
        impl_of(input_), impl_of(target),
        float(delta), _reduction_code(reduction)))
