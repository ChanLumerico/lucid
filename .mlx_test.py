import lucid
import time

t0 = time.time_ns()

x = lucid.ones(4, 4, requires_grad=True) * 2
y = lucid.ones(4, 4, requires_grad=True) * 1

x.to("gpu")
y.to("gpu")

z = (x + y) ** 0.5 @ y
z += z.var(axis=1, keepdims=True)
z.backward(keep_grad=True)
print(x.grad)

t1 = time.time_ns()

print((t1 - t0) / 1e6, "ms")
