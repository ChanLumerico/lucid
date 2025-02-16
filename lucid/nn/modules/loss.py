import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor

from typing import Literal, override


__all__ = ["MSELoss", "BCELoss", "CrossEntropyLoss", "NLLLoss", "HuberLoss"]


_ReductionType = Literal["mean", "sum"]


class _Loss(nn.Module):
    def __init__(self, reduction: _ReductionType | None = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Got an unexpected reduction type: {reduction}.")
        self.reduction = reduction

    @override
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        NotImplemented


class _WeightedLoss(nn.Module):
    def __init__(
        self, weight: Tensor | None = None, reduction: _ReductionType | None = "mean"
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", None}:
            raise ValueError(f"Got an unexpected reduction type: {reduction}.")
        self.reduction = reduction

        self.weight: Tensor | None
        self.register_buffer("weight", weight)

    @override
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        NotImplemented


class MSELoss(_Loss):
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input_, target, reduction=self.reduction)


class BCELoss(_WeightedLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: _ReductionType | None = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__(weight, reduction)
        self.eps = eps

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(
            input_, target, weight=self.weight, reduction=self.reduction, eps=self.eps
        )


class CrossEntropyLoss(_WeightedLoss):
    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: _ReductionType | None = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__(weight, reduction)
        self.eps = eps

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input_, target, weight=self.weight, reduction=self.reduction, eps=self.eps
        )


class NLLLoss(_WeightedLoss):
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input_, target, weight=self.weight, reduction=self.reduction)


class HuberLoss(_Loss):
    def __init__(
        self, reduction: _ReductionType | None = "mean", delta: float = 1.0
    ) -> None:
        super().__init__(reduction)
        self.delta = delta

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(input_, target, delta=self.delta, reduction=self.reduction)
