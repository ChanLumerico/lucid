import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)

x = lucid.random.randn((3, 3), requires_grad=True)
y = lucid.ones((1, 3), requires_grad=True)

z = F.softmax(x + y)
z.backward()

print(z)
print(x.grad)
print(y.grad)
