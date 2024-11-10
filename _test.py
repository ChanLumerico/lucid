import lucid
import numpy as np

lucid.random.seed(42)


x = lucid.random.randn((2, 2), requires_grad=True)
y = lucid.random.randn((2, 2), requires_grad=True)

z = x @ y
z.backward(keep_grad=True)

print(z)

print(z.grad)
print(x.grad)
print(y.grad)
