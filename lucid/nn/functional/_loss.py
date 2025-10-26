from typing import Literal

import lucid
from lucid._tensor import Tensor

_ReductionType = Literal["mean", "sum"]


def _loss_reduction(loss: Tensor, reduction: _ReductionType | None) -> Tensor:
    match reduction:
        case "mean":
            return loss.mean()
        case "sum":
            return loss.sum()
        case None:
            return loss
        case _:
            raise ValueError(
                "Invalid reduction type. Choose 'mean', 'sum', or 'none'.",
            )


def _ignore_index_loss(
    loss: Tensor,
    target_int: Tensor,
    ignore_index: int,
    reduction: _ReductionType | None,
) -> Tensor:
    mask = (target_int != ignore_index).astype(lucid.Float32)
    if reduction is None:
        return loss * mask

    loss_sum = (loss * mask).sum()
    if reduction == "sum":
        return loss_sum

    valid_count = mask.sum()
    if valid_count.item() == 0:
        return lucid.zeros_like(valid_count)

    return loss_sum / valid_count


def mse_loss(
    input_: Tensor, target: Tensor, reduction: _ReductionType | None = "mean"
) -> Tensor:
    loss = (input_ - target) ** 2
    return _loss_reduction(loss, reduction)


def binary_cross_entropy(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    eps: float = 1e-7,
) -> Tensor:
    input_ = lucid.clip(input_, eps, 1 - eps)
    loss = -target * lucid.log(input_) - (1 - target) * lucid.log(1 - input_)

    if weight is not None:
        loss *= weight

    return _loss_reduction(loss, reduction)


def binary_cross_entropy_with_logits(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
) -> Tensor:
    max_val = lucid.maximum(-input_, 0)
    sp = max_val + lucid.log(lucid.exp(-max_val) + lucid.exp(-input_ - max_val))

    if pos_weight is not None:
        coeff = 1 + (pos_weight - 1) * target
        loss = (1 - target) * input_ + coeff * sp
    else:
        loss = lucid.maximum(input_, 0) - input_ * target + sp

    if weight is not None:
        loss *= weight

    return _loss_reduction(loss, reduction)


def cross_entropy(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    eps: float = 1e-7,
    ignore_index: int | None = None,
) -> Tensor:
    exp_logits = lucid.exp(input_ - lucid.max(input_, axis=1, keepdims=True))
    prob = exp_logits / lucid.sum(exp_logits, axis=1, keepdims=True)

    indices = lucid.arange(input_.shape[0], device=input_.device).astype(lucid.Int)
    target_int = target.astype(lucid.Int)

    loss = -lucid.log(prob[indices, target_int] + eps)
    if weight is not None:
        loss *= weight[target_int]

    if ignore_index is not None:
        return _ignore_index_loss(loss, target_int, ignore_index, reduction)

    return _loss_reduction(loss, reduction)


def nll_loss(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    ignore_index: int | None = None,
) -> Tensor:
    target_int = target.astype(lucid.Int)
    n = input_.shape[0]
    idx = lucid.arange(n, device=input_.device).astype(lucid.Int)

    loss = -input_[idx, target_int]
    if weight is not None:
        loss *= weight[target_int]

    if ignore_index is not None:
        return _ignore_index_loss(loss, target_int, ignore_index, reduction)

    return _loss_reduction(loss, reduction)


def huber_loss(
    input_: Tensor,
    target: Tensor,
    delta: float = 1.0,
    reduction: _ReductionType | None = "mean",
) -> Tensor:
    diff = lucid.abs(input_ - target)
    quad = lucid.minimum(diff, delta)
    linear = diff - quad
    loss = 0.5 * quad**2 + delta * linear

    return _loss_reduction(loss, reduction)
