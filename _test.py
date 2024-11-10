import lucid
import numpy as np


x = lucid.tensor(2.0, requires_grad=True)

y = x**2 + 3 * x + 2
y.backward()

print(x.grad)
