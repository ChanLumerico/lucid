"""
lucid.amp: Automatic Mixed Precision utilities.
"""

from lucid.amp.autocast import autocast
from lucid.amp.grad_scaler import GradScaler

__all__ = ["autocast", "GradScaler"]
