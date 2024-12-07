import lucid
import lucid.nn as nn

lucid.random.seed(42)


lin1 = nn.Linear(in_features=3, out_features=6)
lin2 = nn.Linear(in_features=6, out_features=2)

x = lucid.random.randn(1, 3, requires_grad=True)

out = lin1(x)
out = lin2(out)
out = nn.Identity()(out)

out.backward()

print(lin1.weight.grad)
print(lin2.weight.grad)
print(x.grad)
