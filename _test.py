import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)

x = lucid.random.randn((3, 3), requires_grad=True)
y = lucid.ones((3,), requires_grad=True)

z = F.sigmoid(x + y)
z.backward()

print(x.grad)
print(y.grad)
