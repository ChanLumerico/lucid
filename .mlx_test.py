import lucid
import numpy as np
import mlx.core as mx


x = lucid.ones((4, 4), requires_grad=True, device="gpu")

y = x.astype(lucid.Int)
y = y.reshape(-1)
y **= 2

print(y)
y.backward()

print(x.grad)
