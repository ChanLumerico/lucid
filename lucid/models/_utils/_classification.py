"""Image-classification–specific utility modules.

Shared building blocks for classification architectures that do not
belong in any single model family's private namespace.
"""

from typing import cast

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor


class LayerScale(nn.Module):
    """Per-channel learnable scaling applied to a residual branch output.

    Introduced in ConvNeXt (Liu et al., 2022) and used by InceptionNeXt
    and other modern classification architectures.  The parameter ``gamma``
    is initialised to a small constant (typically 1e-6) so that at the
    start of training every block behaves approximately as an identity
    mapping.

    For (B, C)-shaped inputs ``gamma`` is broadcast directly.
    For (B, C, H, W)-shaped inputs ``gamma`` is reshaped to (1, C, 1, 1).

    Args:
        dim:        Number of channels C.
        init_value: Initial value for all entries of ``gamma``.
    """

    def __init__(self, dim: int, init_value: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(lucid.full((dim,), init_value))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if x.ndim == 4:
            # (B, C, H, W) — reshape gamma to broadcast over spatial dims
            g: Tensor = self.gamma.reshape(1, -1, 1, 1)
        else:
            g = self.gamma
        return x * g
