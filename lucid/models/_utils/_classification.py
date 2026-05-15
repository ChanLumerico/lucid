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


class DropPath(nn.Module):
    """Stochastic depth per sample (Huang et al., 2016 "Deep Networks with
    Stochastic Depth").

    Drops the entire residual branch of a sample with probability
    ``drop_prob`` during training; surviving samples are scaled by
    ``1 / (1 - drop_prob)`` so the expected output matches identity.  Acts
    as identity at inference and when ``drop_prob == 0``.

    Used by every modern classification architecture trained with the
    "stochastic depth" recipe — ConvNeXt, Swin, EfficientFormer, MaxViT,
    DeiT, and so on.

    Args:
        drop_prob: Probability of dropping the entire residual branch.  In
            most architectures this scales linearly with depth (deeper
            blocks see higher drop rates).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"DropPath drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample binary mask, broadcast across all non-batch dims.
        shape = (int(x.shape[0]),) + (1,) * (x.ndim - 1)
        random_tensor = lucid.rand(shape, device=x.device.type)
        binary_mask = (random_tensor < keep_prob).float()
        return cast(Tensor, x * binary_mask / keep_prob)
