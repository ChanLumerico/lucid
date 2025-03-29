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


def cross_entropy(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    eps: float = 1e-7,
) -> Tensor:
    exp_logits = lucid.exp(input_ - lucid.max(input_, axis=1, keepdims=True))
    prob = exp_logits / lucid.sum(exp_logits, axis=1, keepdims=True)

    indices = lucid.arange(input_.shape[0], device=input_.device).astype(lucid.Int)
    target_int = target.astype(lucid.Int)

    loss = -lucid.log(prob[indices, target_int] + eps)
    if weight is not None:
        loss *= weight[target_int]

    return _loss_reduction(loss, reduction)


def nll_loss(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
) -> Tensor:
    target_int = target.astype(lucid.Int)
    loss = -input_[:, target_int]
    if weight is not None:
        loss *= weight[target_int]

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
