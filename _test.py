import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)


x = lucid.random.randn((3, 3), requires_grad=True)
y = lucid.var(x, axis=0, keepdims=True)

z = (x @ y.T + y**0.5).dot(y.T).T * lucid.var(x, axis=0, keepdims=True)
z = F.sigmoid(z)

z.backward()

print(z)
print(y.grad)
print(x.grad)
