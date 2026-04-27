"""
lucid.optim.lr_scheduler — learning-rate schedules from the C++ engine.
"""

from __future__ import annotations

from lucid._C import engine as _C_engine

LRScheduler = _C_engine.LRScheduler
LambdaLR = _C_engine.LambdaLR
StepLR = _C_engine.StepLR
MultiStepLR = _C_engine.MultiStepLR
ExponentialLR = _C_engine.ExponentialLR
CosineAnnealingLR = _C_engine.CosineAnnealingLR
ReduceLROnPlateau = _C_engine.ReduceLROnPlateau
CyclicLR = _C_engine.CyclicLR

# NoamScheduler is currently legacy-only; keep the import optional.
try:
    NoamScheduler = _C_engine.NoamScheduler
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
