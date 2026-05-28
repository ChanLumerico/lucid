"""Image-classification–specific utility modules.

Shared building blocks for classification architectures that do not
belong in any single model family's private namespace.
"""

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

    For ``(B, C)``-shaped inputs ``gamma`` is broadcast directly.
    For ``(B, C, H, W)``-shaped inputs ``gamma`` is reshaped to
    ``(1, C, 1, 1)`` so it scales each channel independently across the
    spatial dimensions.

    Parameters
    ----------
    dim : int
        Number of channels :math:`C`.
    init_value : float, optional
        Initial value for every entry of ``gamma``.  Default ``1e-6``
        per the ConvNeXt training recipe — small enough that the block
        starts near identity and large enough that gradients flow.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._classification import LayerScale
    >>> ls = LayerScale(dim=96)
    >>> x = lucid.randn(2, 96, 7, 7)
    >>> ls(x).shape
    (2, 96, 7, 7)

    References
    ----------
    .. [1] Liu et al., *A ConvNet for the 2020s*, CVPR 2022.
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

    Parameters
    ----------
    drop_prob : float, optional
        Probability of dropping the entire residual branch.  In most
        architectures this scales linearly with depth (deeper blocks
        see higher drop rates).  Must satisfy ``0 <= drop_prob < 1``.
        Default ``0.0`` (no dropping).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._classification import DropPath
    >>> dp = DropPath(drop_prob=0.1)
    >>> dp.train()
    >>> x = lucid.randn(8, 96, 7, 7)
    >>> y = dp(x)              # ~10 % of samples zeroed; survivors rescaled
    >>> y.shape
    (8, 96, 7, 7)

    References
    ----------
    .. [1] Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016.
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
        return x * binary_mask / keep_prob
