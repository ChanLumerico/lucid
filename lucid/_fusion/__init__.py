"""
lucid._fusion — Python-level backward-op fusion (currently inactive).

In the legacy code this module declared pairwise fusions
(DoubleNeg, LogExp, ...) consumed by the Python autograd traversal.
The new C++ engine fuses internally, so the Python table is empty.

`match_fusion_table` is kept for any caller that still imports it; it
always returns None (no Python-side fusion candidate).
"""

from __future__ import annotations

from typing import Any


__all__ = ["FusedBackwardOp", "match_fusion_table"]


class FusedBackwardOp:
    """Placeholder retained for source-level compatibility."""

    heuristic_thresh: int = 0


def match_fusion_table(op1: Any, op2: Any) -> type[FusedBackwardOp] | None:
    return None
