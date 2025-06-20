import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _ShapeLike


__all__ = [
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "GlobalResponseNorm",
]


class _NormBase(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            weight_ = lucid.ones((num_features,))
            self.weight = nn.Parameter(weight_)

            bias_ = lucid.zeros((num_features,))
            self.bias = nn.Parameter(bias_)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.running_mean: nn.Buffer
            self.running_var: nn.Buffer

            self.register_buffer("running_mean", lucid.ones((num_features,)))
            self.register_buffer("running_var", lucid.zeros((num_features,)))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero()
            self.running_var.data = lucid.ones((self.num_features,)).data

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.weight = nn.Parameter(lucid.ones_like(self.weight))
            self.bias = nn.Parameter(lucid.zeros_like(self.bias))

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input_: Tensor) -> Tensor:
        return F.batch_norm(
            input_,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum if self.momentum is not None else 0.1,
            self.eps,
        )


class BatchNorm1d(_BatchNorm):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=3)
        return super().forward(input_)


class BatchNorm2d(_BatchNorm):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=4)
        return super().forward(input_)


class BatchNorm3d(_BatchNorm):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=5)
        return super().forward(input_)


class _InstanceNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input_: Tensor) -> Tensor:
        return F.instance_norm(
            input_,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum if self.momentum is not None else 0.1,
            self.eps,
        )


class InstanceNorm1d(_InstanceNorm):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=3)
        return super().forward(input_)


class InstanceNorm2d(_InstanceNorm):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=4)
        return super().forward(input_)


class InstanceNorm3d(_InstanceNorm):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=5)
        return super().forward(input_)


@nn.auto_repr("normalized_shape", "eps", "elementwise_affine")
class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: _ShapeLike | int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            weight_ = lucid.ones(self.normalized_shape)
            self.weight = nn.Parameter(weight_)

            if bias:
                bias_ = lucid.zeros(self.normalized_shape)
                self.bias = nn.Parameter(bias_)
            else:
                self.bias = None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input_: Tensor) -> Tensor:
        return F.layer_norm(
            input_, self.normalized_shape, self.weight, self.bias, self.eps
        )


@nn.auto_repr("channels", "eps")
class GlobalResponseNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(lucid.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(lucid.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, input_: Tensor) -> Tensor:
        return F.global_response_norm(input_, self.gamma, self.beta, self.eps)
