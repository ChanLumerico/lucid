import lucid
import numpy as np
import mlx.core as mx


x = lucid.random.randn(1, 3, 4, 4, requires_grad=True).to("gpu")

y = 2 / x + x
y.backward()

print(y)
print(y.dtype)
print(x.grad.dtype)
