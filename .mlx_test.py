import lucid
import numpy as np
import mlx.core as mx


x = lucid.ones((4, 3), requires_grad=True, device="gpu")

y, z = lucid.chunk(x, 2)

print(y.shape, z.shape)
# y.backward()
z.backward()

print(x.grad)
