import lucid
import time

t0 = time.time_ns()

x = lucid.ones(200, 300, requires_grad=True)
y = lucid.ones(300, 400, requires_grad=True)

x.to("gpu")
y.to("gpu")

z = x @ -y
z = z.var()

print(z.dtype)
z.backward()

t = time.time_ns()
print(f"{(t - t0) / 1e6} ms")
