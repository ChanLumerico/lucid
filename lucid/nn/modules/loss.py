"""
Loss function modules.
"""

from typing import Any
from lucid.nn.module import Module
from lucid.nn.functional.loss import (
    mse_loss,
    l1_loss,
    cross_entropy,
    nll_loss,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    huber_loss,
    smooth_l1_loss,
    kl_div,
)


class MSELoss(Module):
    """Mean squared error loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Any, target: Any) -> Any:
        return mse_loss(x, target, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class L1Loss(Module):
    """Mean absolute error loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Any, target: Any) -> Any:
        return l1_loss(x, target, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification."""

    def __init__(
        self,
        weight: Any = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, x: Any, target: Any) -> Any:
        return cross_entropy(
            x,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, reduction={self.reduction!r}, "
            f"label_smoothing={self.label_smoothing}"
        )


class NLLLoss(Module):
    """Negative log-likelihood loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Any, target: Any) -> Any:
        return nll_loss(x, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class BCELoss(Module):
    """Binary cross-entropy loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Any, target: Any) -> Any:
        return binary_cross_entropy(x, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class BCEWithLogitsLoss(Module):
    """BCE with logits (sigmoid + BCE combined)."""

    def __init__(self, reduction: str = "mean", pos_weight: Any = None) -> None:
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, x: Any, target: Any) -> Any:
        return binary_cross_entropy_with_logits(
            x, target, pos_weight=self.pos_weight, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class HuberLoss(Module):
    """Huber loss."""

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, x: Any, target: Any) -> Any:
        return huber_loss(x, target, self.delta, self.reduction)

    def extra_repr(self) -> str:
        return f"delta={self.delta}, reduction={self.reduction!r}"


class SmoothL1Loss(Module):
    """Smooth L1 loss (Huber loss with delta=beta)."""

    def __init__(self, reduction: str = "mean", beta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, x: Any, target: Any) -> Any:
        return smooth_l1_loss(x, target, beta=self.beta, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, reduction={self.reduction!r}"


class KLDivLoss(Module):
    """Kullback-Leibler divergence loss."""

    def __init__(self, reduction: str = "mean", log_target: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, x: Any, target: Any) -> Any:
        return kl_div(x, target, reduction=self.reduction, log_target=self.log_target)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}, log_target={self.log_target}"
