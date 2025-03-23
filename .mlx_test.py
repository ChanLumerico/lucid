import lucid

lucid.random.seed(42)


x = lucid.random.randn(3, 3, requires_grad=True, device="cpu")

u, v = lucid.linalg.eig(x)
u.backward()

print(u)
print(v)
