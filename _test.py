import lucid
import lucid.nn as nn

import numpy as np

lucid.random.seed(42)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(np.random.randn(in_features, out_features))
        self.bias = nn.Parameter(np.zeros(out_features))

    def forward(self, x):
        return lucid.dot(x, self.weights) + self.bias


x = lucid.random.randn((5, 2), requires_grad=True)

lin = Linear(in_features=2, out_features=4)

out = lin.forward(x)

print(out)
