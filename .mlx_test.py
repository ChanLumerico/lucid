import lucid
import time

t0 = time.time_ns()

x = lucid.ones(4, 4, requires_grad=True) * 2
y = lucid.ones(4, 4, requires_grad=True) * 3

x.to("gpu")
y.to("gpu")

z = x / y
z.backward(keep_grad=True)
print(x.grad)

t1 = time.time_ns()

print((t1 - t0) / 1e6, "ms")
