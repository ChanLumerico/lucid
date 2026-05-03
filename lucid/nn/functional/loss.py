"""
nn.functional loss functions.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_REDUCTION_MAP = {"none": 0, "mean": 1, "sum": 2}


def mse_loss(
    x: "Tensor", target: "Tensor", reduction: str = "mean"
) -> "Tensor":
    """Mean squared error loss."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.mse_loss(_unwrap(x), _unwrap(target), red))


def l1_loss(
    x: "Tensor", target: "Tensor", reduction: str = "mean"
) -> "Tensor":
    """Mean absolute error loss."""
    diff = _C_engine.abs(_C_engine.sub(_unwrap(x), _unwrap(target)))
    if reduction == "mean":
        return _wrap(_C_engine.mean(diff, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(diff, [], False))
    return _wrap(diff)


def smooth_l1_loss(
    x: "Tensor", target: "Tensor", beta: float = 1.0, reduction: str = "mean"
) -> "Tensor":
    """Smooth L1 (Huber) loss."""
    return huber_loss(x, target, delta=beta, reduction=reduction)


def huber_loss(
    x: "Tensor", target: "Tensor", delta: float = 1.0, reduction: str = "mean"
) -> "Tensor":
    """Huber loss."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.huber_loss(_unwrap(x), _unwrap(target), delta, red))


def cross_entropy(
    x: "Tensor",
    target: "Tensor",
    weight: "Tensor | None" = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> "Tensor":
    """Cross-entropy loss for multi-class classification."""
    red = _REDUCTION_MAP.get(reduction, 1)
    w = _unwrap(weight) if weight is not None else None
    return _wrap(
        _C_engine.nn.cross_entropy_loss(_unwrap(x), _unwrap(target), red)
    )


def nll_loss(
    x: "Tensor",
    target: "Tensor",
    weight: "Tensor | None" = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> "Tensor":
    """Negative log-likelihood loss."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.nll_loss(_unwrap(x), _unwrap(target), red))


def binary_cross_entropy(
    x: "Tensor",
    target: "Tensor",
    weight: "Tensor | None" = None,
    reduction: str = "mean",
) -> "Tensor":
    """Binary cross-entropy loss."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.bce_loss(_unwrap(x), _unwrap(target), red))


def binary_cross_entropy_with_logits(
    x: "Tensor",
    target: "Tensor",
    weight: "Tensor | None" = None,
    pos_weight: "Tensor | None" = None,
    reduction: str = "mean",
) -> "Tensor":
    """BCE with logits loss (combines sigmoid + BCE for numerical stability)."""
    red = _REDUCTION_MAP.get(reduction, 1)
    return _wrap(_C_engine.nn.bce_with_logits(_unwrap(x), _unwrap(target), red))


def kl_div(
    x: "Tensor",
    target: "Tensor",
    size_average: bool | None = None,
    reduction: str = "mean",
    log_target: bool = False,
) -> "Tensor":
    """Kullback-Leibler divergence."""
    from lucid._C import engine as E
    import numpy as np
    from lucid._dispatch import _wrap, _unwrap
    # KL(P||Q) = sum(P * (log P - log Q))
    if log_target:
        diff = _C_engine.sub(_unwrap(target), _unwrap(x))
        kl = _C_engine.mul(_C_engine.exp(_unwrap(target)), diff)
    else:
        log_x = _C_engine.log(_unwrap(x))
        diff = _C_engine.sub(_C_engine.log(_unwrap(target)), log_x)
        kl = _C_engine.mul(_unwrap(target), diff)
    if reduction == "mean":
        return _wrap(_C_engine.mean(kl, [], False))
    if reduction == "sum":
        return _wrap(_C_engine.sum(kl, [], False))
    return _wrap(kl)
