"""
Loss function modules.
"""

from typing import Any
from lucid.nn.module import Module
# F imported lazily inside forward()


class MSELoss(Module):
    """Mean squared error loss."""
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, x: Any, target: Any) -> Any:
        from lucid.nn import functional as F
        return F.mse_loss(x, target, self.reduction)


class L1Loss(Module):
    """Mean absolute error loss."""
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, x: Any, target: Any) -> Any:
        from lucid.nn import functional as F
        return F.l1_loss(x, target, self.reduction)


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
        from lucid.nn import functional as F
        return F.cross_entropy(
            x, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class NLLLoss(Module):
    """Negative log-likelihood loss."""
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, x: Any, target: Any) -> Any:
        from lucid.nn import functional as F
        return F.nll_loss(x, target, reduction=self.reduction)


class BCELoss(Module):
    """Binary cross-entropy loss."""
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, x: Any, target: Any) -> Any:
        from lucid.nn import functional as F
        return F.binary_cross_entropy(x, target, reduction=self.reduction)


class BCEWithLogitsLoss(Module):
    """BCE with logits (sigmoid + BCE combined)."""
    def __init__(self, reduction: str = "mean", pos_weight: Any = None) -> None:
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight
    def forward(self, x: Any, target: Any) -> Any:
        from lucid.nn import functional as F
        return F.binary_cross_entropy_with_logits(
            x, target, pos_weight=self.pos_weight, reduction=self.reduction
        )


class HuberLoss(Module):
    """Huber loss."""
    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.delta = delta
    def forward(self, x: Any, target: Any) -> Any:
        from lucid.nn import functional as F
        return F.huber_loss(x, target, self.delta, self.reduction)
