import lucid
import numpy as np


x = lucid.tensor(np.random.randn(3, 3), requires_grad=True)

z = lucid.trace(x) * 2

z.backward(keep_grad=True)

print(z)
print(x.grad)