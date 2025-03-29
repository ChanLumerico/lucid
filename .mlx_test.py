import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

import numpy as np
import mlx.core as mx

lucid.random.seed(42)


x = lucid.random.randn(10, 10, requires_grad=True, device="gpu")

y = F.layer_norm(x, (10, 10))
y.astype(lucid.Int16)

y.backward()

print(y.dtype.__repr__())
print(y.device)
print(y.shape)
