import lucid
import numpy as np


x = lucid.tensor([1, 2], requires_grad=True)
y = lucid.tensor([3, 4], requires_grad=True)

z = x.dot(y)

z.backward(keep_grad=True)

print(z)

print(y.grad)
print(x.grad)
