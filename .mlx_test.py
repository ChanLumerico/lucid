import lucid
import numpy as np
import mlx.core as mx


x = lucid.Tensor([1, 2, 3], requires_grad=True, device="gpu")

y = x + x
y.backward()
print(y)
print(x.grad)
