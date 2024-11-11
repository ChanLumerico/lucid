import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["Linear"]


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight_ = nn.Parameter(
            lucid.random.randn((out_features, in_features)) * 0.01
        )
        self.bias_ = nn.Parameter(lucid.zeros((1, out_features))) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight_, self.bias_)
