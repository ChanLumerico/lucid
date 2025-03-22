import lucid
import numpy as np
import mlx.core as mx


# fmt: off
x = lucid.Tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]], requires_grad=True, device="gpu"
)

y = x.roll(1)
y.backward()

print(y)
print(x.grad)
