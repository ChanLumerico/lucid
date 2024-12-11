import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)


x = lucid.random.randn(100, 3, 28, 28, requires_grad=True)
y = lucid.tensor([0, 1, 2, 3] * 25)

seq = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=3),
    nn.BatchNorm2d(6),
    nn.ReLU(),
    nn.Conv2d(6, 12, kernel_size=3),
    nn.BatchNorm2d(12),
    nn.ReLU(),
)

import time

t0 = time.time_ns()

out = seq(x)
out.backward()

print(f"{(time.time_ns() - t0) / 1e9} sec\n")

print(out.shape if out.ndim > 0 else out.item())
print(x.grad.shape)
