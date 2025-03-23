import lucid

lucid.random.seed(42)


x = lucid.random.randn(3, 3, requires_grad=True, device="gpu")

u, v = lucid.linalg.eig(x)
print(u.shape, v.shape)

u.backward()
v.backward()

print(x.grad.shape)
