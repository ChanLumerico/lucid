import lucid

lucid.random.seed(42)


x = lucid.random.randn(4, 4, requires_grad=True, device="gpu")

y = lucid.einops.repeat(x, "i j -> i j k", k=2)
y.backward()

print(y.shape)
print(x.grad.shape)
