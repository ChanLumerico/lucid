import lucid
import time

t0 = time.time_ns()

x = lucid.ones(200, 200, requires_grad=True)
y = lucid.ones(200, 200, requires_grad=True)

x.to("gpu")
y.to("gpu")

z = x - y
z.backward()
print(x.grad.shape)

t1 = time.time_ns()

print((t1 - t0) / 1e6, "ms")
