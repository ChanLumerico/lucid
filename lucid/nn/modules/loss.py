import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor

from typing import Literal, override


__all__ = ["MSELoss"]


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


class _WeightedLoss(nn.Module): ...


class MSELoss(_Loss):
    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input_, target, reduction=self.reduction)
