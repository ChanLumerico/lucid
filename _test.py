import lucid
import lucid.nn as nn

x = lucid.random.randn((3, 4), requires_grad=True)
y = lucid.random.randn((3, 4), requires_grad=True)

z = lucid.trace(x @ y.T)
z.backward()

print(z)

print(x.grad)
print(y.grad)
