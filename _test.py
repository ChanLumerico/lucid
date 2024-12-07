import lucid
import lucid.nn as nn


# Example custom modules
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        weight_ = lucid.random.randn(in_features, out_features)
        bias_ = lucid.zeros((1, out_features))

        self.weight = nn.Parameter(weight_)
        self.bias = nn.Parameter(bias_)

    def forward(self, x):
        return x @ self.weight + self.bias


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # Register two Linear submodules
        self.lin1 = Linear(input_size, hidden_size)
        self.lin2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x


# Testing parameters() and modules()
if __name__ == "__main__":
    model = MLP(input_size=5, hidden_size=10, output_size=3)

    print("Modules in the model:")
    for m in model.modules():
        print(m)

    print("\nParameters in the model:")
    for p in model.parameters():
        print(p.shape)
