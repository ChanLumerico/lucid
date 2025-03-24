import lucid

lucid.random.seed(42)


x = lucid.random.randn(3, 3, requires_grad=True, device="gpu")

y = lucid.linalg.pinv(x)
y.backward()

print(y)
print(x.grad)
