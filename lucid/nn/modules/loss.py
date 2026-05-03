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
    triplet_margin_loss,
    cosine_embedding_loss,
    margin_ranking_loss,
    hinge_embedding_loss,
    poisson_nll_loss,
    gaussian_nll_loss,
    ctc_loss,
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


class TripletMarginLoss(Module):
    """Triplet margin loss."""

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor: Any, positive: Any, negative: Any) -> Any:
        return triplet_margin_loss(
            anchor, positive, negative,
            margin=self.margin, p=self.p, eps=self.eps,
            swap=self.swap, reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"margin={self.margin}, p={self.p}, reduction={self.reduction!r}"


class CosineEmbeddingLoss(Module):
    """Cosine embedding loss."""

    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1: Any, x2: Any, y: Any) -> Any:
        return cosine_embedding_loss(x1, x2, y, margin=self.margin, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"margin={self.margin}, reduction={self.reduction!r}"


class MarginRankingLoss(Module):
    """Margin ranking loss: max(0, -y*(x1-x2) + margin)."""

    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1: Any, x2: Any, y: Any) -> Any:
        return margin_ranking_loss(x1, x2, y, margin=self.margin, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"margin={self.margin}, reduction={self.reduction!r}"


class HingeEmbeddingLoss(Module):
    """Hinge embedding loss."""

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x: Any, y: Any) -> Any:
        return hinge_embedding_loss(x, y, margin=self.margin, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"margin={self.margin}, reduction={self.reduction!r}"


class PoissonNLLLoss(Module):
    """Poisson negative log-likelihood loss."""

    def __init__(
        self,
        log_input: bool = True,
        full: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.log_input = log_input
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: Any, target: Any) -> Any:
        return poisson_nll_loss(
            x, target, log_input=self.log_input, full=self.full,
            eps=self.eps, reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"log_input={self.log_input}, reduction={self.reduction!r}"


class GaussianNLLLoss(Module):
    """Gaussian negative log-likelihood loss."""

    def __init__(
        self,
        full: bool = False,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: Any, target: Any, var: Any) -> Any:
        return gaussian_nll_loss(
            x, target, var, full=self.full, eps=self.eps, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class CTCLoss(Module):
    """Connectionist Temporal Classification loss."""

    def __init__(
        self,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False,
    ) -> None:
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self, log_probs: Any, targets: Any, input_lengths: Any, target_lengths: Any
    ) -> Any:
        return ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=self.blank, reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )

    def extra_repr(self) -> str:
        return f"blank={self.blank}, reduction={self.reduction!r}"
