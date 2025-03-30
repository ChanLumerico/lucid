import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

import numpy as np
import mlx.core as mx

import time

lucid.random.seed(42)

device = "gpu"


x = lucid.random.randn(10, 64, requires_grad=True, device=device)
w = lucid.random.randn(64, 128, requires_grad=True, device=device)

y = F.embedding(x, w)
y.backward()

print(y.shape)
print(y.dtype.__repr__())
print(y.data.dtype)
print(y.device)

print(w.grad.shape if w.grad is not None else None)
