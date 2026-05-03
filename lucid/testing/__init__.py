"""
lucid.testing — Numerical verification utilities.

Provides tools for testing tensor operations, gradient correctness,
and numerical accuracy. Analogous to ``torch.testing``.
"""

from lucid.testing._comparison import assert_close

__all__ = ["assert_close"]
