import math

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

        weight_ = lucid.empty(out_features, in_features)
        self.weight = nn.Parameter(weight_)

        if bias:
            bias_ = lucid.empty(out_features)
            self.bias = nn.Parameter(bias_)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform(self.weight)

        if self.bias is not None:
            fan_in, _ = nn.init._dist._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform(self.bias, -bound, bound)

    def forward(self, input_: Tensor) -> Tensor:
        return F.linear(input_, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class Bilinear(nn.Module):
    def __init__(
        self, in1_features: int, in2_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        weight_ = lucid.empty(out_features, in1_features, in2_features)
        self.weight = nn.Parameter(weight_)

        if bias:
            bias_ = lucid.empty(out_features)
            self.bias = nn.Parameter(bias_)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.shape[1])
        nn.init.uniform(self.weight, -bound, bound)

        if self.bias is not None:
            nn.init.uniform(self.bias, -bound, bound)

    def forward(self, input_1: Tensor, input_2: Tensor) -> Tensor:
        return F.bilinear(input_1, input_2, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"{self.in1_features}, {self.in2_features}, {self.out_features}, "
            f"bias={self.bias is not None}"
        )
