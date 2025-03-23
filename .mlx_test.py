import lucid
import numpy as np
import mlx.core as mx


x = lucid.random.bernoulli([0.3, 0.4, 0.3], device="gpu")

a = lucid.random.randn(3, requires_grad=True, device="gpu")

print(x)
print(a[x])
