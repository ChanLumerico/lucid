import lucid


x = lucid.arange(5, requires_grad=True)
y = -lucid.arange(5, requires_grad=True)

a, b = lucid.meshgrid(x, y, indexing="xy")

c = a * b
c.backward()

print(a.grad)
print(b.grad)
