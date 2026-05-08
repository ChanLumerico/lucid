"""
Loss function modules.
"""

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from collections.abc import Callable

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
    multi_margin_loss,
    multilabel_margin_loss,
    soft_margin_loss,
    multilabel_soft_margin_loss,
)


class MSELoss(Module):
    """Mean squared error loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return mse_loss(x, target, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class L1Loss(Module):
    """Mean absolute error loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return l1_loss(x, target, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification."""

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
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
    """Negative log-likelihood loss.

    Parameters
    ----------
    weight : (C,) optional per-class weight tensor.
    ignore_index : samples with this class are excluded from the loss.
    reduction : 'none' | 'mean' | 'sum'.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight: Tensor | None = weight
        self.ignore_index: int = ignore_index
        self.reduction: str = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return nll_loss(
            x,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        s: str = f"reduction={self.reduction!r}"
        if self.ignore_index != -100:
            s += f", ignore_index={self.ignore_index}"
        return s


class BCELoss(Module):
    """Binary cross-entropy loss.

    Parameters
    ----------
    weight : optional element-wise weight tensor (broadcast over input/target).
    reduction : 'none' | 'mean' | 'sum'.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight: Tensor | None = weight
        self.reduction: str = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy(
            x, target, weight=self.weight, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class BCEWithLogitsLoss(Module):
    """BCE with logits — combined sigmoid + BCE for numerical stability.

    Parameters
    ----------
    weight : optional element-wise weight (broadcast over input/target).
    reduction : 'none' | 'mean' | 'sum'.
    pos_weight : optional positive-class weight (broadcast over the
        trailing dim) to up-weight the positive examples.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weight: Tensor | None = weight
        self.reduction: str = reduction
        self.pos_weight: Tensor | None = pos_weight

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(
            x,
            target,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class HuberLoss(Module):
    """Huber loss."""

    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.delta = delta

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return huber_loss(x, target, self.delta, self.reduction)

    def extra_repr(self) -> str:
        return f"delta={self.delta}, reduction={self.reduction!r}"


class SmoothL1Loss(Module):
    """Smooth L1 loss (Huber loss with delta=beta)."""

    def __init__(self, reduction: str = "mean", beta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return smooth_l1_loss(x, target, beta=self.beta, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, reduction={self.reduction!r}"


class KLDivLoss(Module):
    """Kullback-Leibler divergence loss."""

    def __init__(self, reduction: str = "mean", log_target: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
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

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return triplet_margin_loss(
            anchor,
            positive,
            negative,
            margin=self.margin,
            p=self.p,
            eps=self.eps,
            swap=self.swap,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"margin={self.margin}, p={self.p}, reduction={self.reduction!r}"


class CosineEmbeddingLoss(Module):
    """Cosine embedding loss."""

    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1: Tensor, x2: Tensor, y: Tensor) -> Tensor:
        return cosine_embedding_loss(
            x1, x2, y, margin=self.margin, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        return f"margin={self.margin}, reduction={self.reduction!r}"


class MarginRankingLoss(Module):
    """Margin ranking loss: max(0, -y*(x1-x2) + margin)."""

    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x1: Tensor, x2: Tensor, y: Tensor) -> Tensor:
        return margin_ranking_loss(
            x1, x2, y, margin=self.margin, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        return f"margin={self.margin}, reduction={self.reduction!r}"


class HingeEmbeddingLoss(Module):
    """Hinge embedding loss."""

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
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

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return poisson_nll_loss(
            x,
            target,
            log_input=self.log_input,
            full=self.full,
            eps=self.eps,
            reduction=self.reduction,
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

    def forward(self, x: Tensor, target: Tensor, var: Tensor) -> Tensor:
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
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        return ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
        )

    def extra_repr(self) -> str:
        return f"blank={self.blank}, reduction={self.reduction!r}"


class MultiMarginLoss(Module):
    """Multi-class margin (SVM-style) loss."""

    def __init__(
        self,
        p: int = 1,
        margin: float = 1.0,
        weight: "Tensor | None" = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return multi_margin_loss(
            x,
            target,
            p=self.p,
            margin=self.margin,
            weight=self.weight,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f"p={self.p}, margin={self.margin}, reduction={self.reduction!r}"


class MultilabelMarginLoss(Module):
    """Multi-label margin loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return multilabel_margin_loss(x, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


# CamelCase alias for parity with the reference framework's
# ``MultiLabelMarginLoss`` (capital ``L``).  Kept as a subclass rather
# than a simple ``= MultilabelMarginLoss`` so ``__name__`` and
# ``__repr__`` carry the canonical name.
class MultiLabelMarginLoss(MultilabelMarginLoss):
    """CamelCase alias for :class:`MultilabelMarginLoss` — provided so
    ``nn.MultiLabelMarginLoss`` (the reference framework's spelling)
    resolves to the same implementation.  No behavioural difference."""


# ── P4 fill: SoftMargin / MultiLabelSoftMargin / TripletMarginWithDistance ──


class SoftMarginLoss(Module):
    """``log(1 + exp(-target · input))`` averaged per element."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return soft_margin_loss(x, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class MultiLabelSoftMarginLoss(Module):
    """Element-wise binary cross-entropy with logits, averaged across the
    class dimension — the multi-label soft-margin form."""

    def __init__(self, weight: Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return multilabel_soft_margin_loss(
            x, target, weight=self.weight, reduction=self.reduction
        )

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class TripletMarginWithDistanceLoss(Module):
    """Triplet margin loss with a user-supplied distance function.

    Generalises :class:`TripletMarginLoss` — the caller passes a callable
    ``distance_function(x, y) -> Tensor`` instead of an L_p norm.  Falls
    back to L₂ when none is provided, matching the reference framework.
    """

    def __init__(
        self,
        distance_function: Callable[[Tensor, Tensor], Tensor] | None = None,
        margin: float = 1.0,
        swap: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if distance_function is None:
            from lucid.nn.functional.activations import pairwise_distance

            def _default(a: Tensor, b: Tensor) -> Tensor:
                return pairwise_distance(a, b, p=2.0)

            self.distance_function: Callable[[Tensor, Tensor], Tensor] = _default
        else:
            self.distance_function = distance_function
        self.margin = margin
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        # Delegate to the functional implementation so the F. and nn.
        # surfaces stay byte-equivalent.
        from lucid.nn.functional.loss import triplet_margin_with_distance_loss

        return triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return (
            f"margin={self.margin}, swap={self.swap}, " f"reduction={self.reduction!r}"
        )
