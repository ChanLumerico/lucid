import lucid
import numpy as np
import mlx.core as mx


x = lucid.Tensor([1, 2, 3, 4], requires_grad=True, device="cpu")

y, z = lucid.meshgrid(x, x)

w = y + z
w.backward()
