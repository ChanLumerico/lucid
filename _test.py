import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)

x = lucid.tensor([[1, 2], [3, 4]], requires_grad=True)
y = x.ravel() ** 2

y.backward()

print(y)
print(x.grad)
