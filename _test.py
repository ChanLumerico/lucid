import lucid
import numpy as np


x = lucid.tensor([1, 2], requires_grad=True)
y = lucid.tensor([3, 4], requires_grad=True)

z = x @ y

print(z)
z.backward(keep_grad=True)

print(z.grad)
print(y.grad)
print(x.grad)
