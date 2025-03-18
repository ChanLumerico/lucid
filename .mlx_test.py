import lucid
import numpy as np
import mlx.core as mx


x = lucid.ones((4, 4), requires_grad=True, device="gpu")

y = x.astype(lucid.Int)
y **= 2

y.backward()

print(x.grad)
