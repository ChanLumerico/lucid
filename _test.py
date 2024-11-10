import lucid
import lucid.nn as nn

lucid.random.seed(42)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(lucid.random.randn((in_features, out_features)))
        self.bias = nn.Parameter(lucid.random.randn((out_features,)))

    def forward(self, x):
        return lucid.dot(x, self.weights) + self.bias


lin = Linear(in_features=2, out_features=4)

x = lucid.random.randn((3, 2), requires_grad=True)
y = lin(x)

y.backward()

print(x.grad)
