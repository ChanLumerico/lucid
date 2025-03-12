import lucid
import numpy as np
import mlx.core as mx


x = lucid.ones(2, 3, requires_grad=True).to("gpu")
y = lucid.ones(3, 4, requires_grad=True).to("gpu")


z = x @ y
z.backward()
