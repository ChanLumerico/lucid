import lucid
import numpy as np
import mlx.core as mx


x = lucid.ones((4, 4), requires_grad=True, device="gpu").astype(lucid.Int)

y = lucid.repeat(x, 3, axis=0).flatten()

print("y-shape:", y.shape)
y.backward()

print(x.grad)
