"""
Normalization modules.
"""

from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import ones, zeros
import lucid.nn.init as init
from lucid.nn.functional.normalization import (
    layer_norm,
    rms_norm,
    group_norm,
    batch_norm,
)


class LayerNorm(Module):
    """Layer normalization."""

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: tuple[int, ...] = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight: Parameter | None = Parameter(
                ones(*self.normalized_shape, dtype=dtype, device=device)
            )
            self.bias: Parameter | None = Parameter(
                zeros(*self.normalized_shape, dtype=dtype, device=device)
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Any) -> Any:
        return layer_norm(
            x, list(self.normalized_shape), self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class RMSNorm(Module):
    """RMS normalization."""

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-8,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(
            ones(*self.normalized_shape, dtype=dtype, device=device)
        )

    def forward(self, x: Any) -> Any:
        return rms_norm(x, list(self.normalized_shape), self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class GroupNorm(Module):
    """Group normalization."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight: Parameter | None = Parameter(
                ones(num_channels, dtype=dtype, device=device)
            )
            self.bias: Parameter | None = Parameter(
                zeros(num_channels, dtype=dtype, device=device)
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Any) -> Any:
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"


class _BatchNormBase(Module):
    """Common implementation for BatchNorm1d/2d/3d."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight: Parameter | None = Parameter(
                ones(num_features, dtype=dtype, device=device)
            )
            self.bias: Parameter | None = Parameter(
                zeros(num_features, dtype=dtype, device=device)
            )
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.register_buffer(
                "running_mean", zeros(num_features, dtype=dtype, device=device)
            )
            self.register_buffer(
                "running_var", ones(num_features, dtype=dtype, device=device)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def forward(self, x: Any) -> Any:
        return batch_norm(
            x,
            self._buffers.get("running_mean"),
            self._buffers.get("running_var"),
            self.weight,
            self.bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm1d(_BatchNormBase):
    """Batch normalization for 2D or 3D input (N, C) or (N, C, L)."""


class BatchNorm2d(_BatchNormBase):
    """Batch normalization for 4D input (N, C, H, W)."""


class BatchNorm3d(_BatchNormBase):
    """Batch normalization for 5D input (N, C, D, H, W)."""


class InstanceNorm1d(_BatchNormBase):
    """Instance normalization for 3D input."""


class InstanceNorm2d(_BatchNormBase):
    """Instance normalization for 4D input."""


class InstanceNorm3d(_BatchNormBase):
    """Instance normalization for 5D input."""
