import lucid
import lucid.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weights_ = nn.Parameter(lucid.random.randn((in_features, out_features)))
        self.bias = bias

        if self.bias:
            self.bias_ = nn.Parameter(lucid.random.randn((1, out_features)))

    def forward(self, x):
        return (
            lucid.dot(x, self.weights_) + self.bias_
            if self.bias
            else lucid.dot(x, self.weights_)
        )
