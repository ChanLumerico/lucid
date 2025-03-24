import lucid

lucid.random.seed(42)


x = lucid.random.randn(3, 3, requires_grad=True, device="gpu")
y = lucid.linalg.norm(x, axis=(-1, -2))

y.backward()

print(x.grad)
