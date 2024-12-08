import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d"]


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
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.runnung_mean = lucid.zeros((num_features,))
            self.running_var = lucid.ones((num_features,))
        else:
            self.runnung_mean = None
            self.running_var = None

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.runnung_mean.zero()
            self.running_var.data = lucid.ones((self.num_features,)).data

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.weight = nn.Parameter(lucid.ones_like(self.weight))
            self.bias = nn.Parameter(lucid.zeros_like(self.bias))


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
        if self.track_running_stats:
            running_mean = self.runnung_mean
            running_var = self.running_var
        else:
            running_mean = None
            running_var = None

        training_mode = self.training

        return F.batch_norm(
            input_,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            training_mode,
            self.momentum if self.momentum is not None else 0.1,
            self.eps,
        )
