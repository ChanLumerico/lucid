import lucid
import numpy as np
import mlx.core as mx


x = lucid.ones(200, 300, requires_grad=True).to("gpu")
y = lucid.ones(300, 400, requires_grad=True).to("gpu")


z = x @ y
z.backward()
