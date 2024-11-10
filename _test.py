import lucid
import numpy as np


x = lucid.tensor([[1, 2], [3, 4]], requires_grad=True)
y = lucid.tensor([[5, 6], [7, 8]], requires_grad=True)

z = (x @ y).sum(axis=1, keepdims=True)

z.backward(keep_grad=True)

print(z)

print(z.grad)
print(y.grad)
print(x.grad)
