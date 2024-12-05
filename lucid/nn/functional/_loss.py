from typing import Literal, LiteralString

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


def cross_entropy(  # TODO: Need to be inspected; shape mismatch during backward
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
    eps: float = 1e-7,
) -> Tensor:
    input_ = input_ - lucid.max(input_, axis=1, keepdims=True)
    log_probs = input_ - lucid.log(lucid.exp(input_).sum(axis=1, keepdims=True) + eps)

    loss = -log_probs[:, target.data.astype(int)]
    if weight is not None:
        loss *= weight[target.data.astype(int)]

    print(input_.shape, log_probs.shape, loss.shape)

    return _loss_reduction(loss, reduction)


def nll_loss(
    input_: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    reduction: _ReductionType | None = "mean",
) -> Tensor:
    loss = -input_[:, target.data.astype(int)]
    if weight is not None:
        loss *= weight[target.data.astype(int)]

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
