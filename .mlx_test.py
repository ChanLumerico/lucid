import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

import numpy as np
import mlx.core as mx

import time

lucid.random.seed(42)

device = "gpu"


x = lucid.random.randn(100, 3, 224, 224, requires_grad=True, device=device)

conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
conv.to(device)

t0 = time.time_ns()

y = conv(x)
y.backward()

t1 = time.time_ns()

print(f"{(t1 - t0) / 1e6} ms")

# NOTE
# CPU: 749.361278 ms
# GPU:   2.361677 ms

# print(y.shape)
# print(y.dtype.__repr__())
# print(y.data.dtype)
# print(y.device)

# print(x.grad.shape)
