"""
lucid.optim.lr_scheduler — learning-rate schedules from the C++ engine.
"""

from __future__ import annotations

from lucid._C import engine as _eng

LRScheduler = _eng.LRScheduler
LambdaLR = _eng.LambdaLR
StepLR = _eng.StepLR
MultiStepLR = _eng.MultiStepLR
ExponentialLR = _eng.ExponentialLR
CosineAnnealingLR = _eng.CosineAnnealingLR
ReduceLROnPlateau = _eng.ReduceLROnPlateau
CyclicLR = _eng.CyclicLR

# NoamScheduler is currently legacy-only; keep the import optional.
try:
    NoamScheduler = _eng.NoamScheduler
except AttributeError:  # pragma: no cover
    pass

__all__ = [
    "LRScheduler",
    "LambdaLR",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "CyclicLR",
]
