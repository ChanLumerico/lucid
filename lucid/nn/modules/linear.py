import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["Identity", "Linear", "Bilinear"]


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return input_


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight_ = lucid.random.rand(out_features, in_features)
        self.weight = nn.Parameter(weight_ * 0.01)

        if bias:
            bias_ = lucid.zeros((1, out_features))
            self.bias = nn.Parameter(bias_)
        else:
            self.bias = None

    def forward(self, input_: Tensor) -> Tensor:
        return F.linear(input_, self.weight, self.bias)


class Bilinear(nn.Module):
    def __init__(
        self, in1_features: int, in2_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        weight_ = lucid.random.rand(out_features, in1_features, in2_features)
        self.weight = nn.Parameter(weight_ * 0.01)

        if bias:
            bias_ = lucid.zeros((1, out_features))
            self.bias = nn.Parameter(bias_)
        else:
            self.bias = None

    def forward(self, input_1: Tensor, input_2: Tensor) -> Tensor:
        return F.bilinear(input_1, input_2, self.weight, self.bias)
