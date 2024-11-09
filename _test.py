import lucid
import numpy as np


x = lucid.tensor(np.random.randn(2, 2), requires_grad=True)
y = lucid.tensor(np.random.randn(2, 2), requires_grad=True)

z = (x @ y + lucid.eye(2)).T

print(lucid.diag([3, 2, 1]))

z.backward()

print(f"z: {z}")
print(f"dz/dx: {x.grad}")
print(f"dz/dy: {y.grad}")
