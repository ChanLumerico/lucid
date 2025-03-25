import lucid

lucid.random.seed(42)


x = lucid.random.randn(2, 3, 4, 5, requires_grad=True, device="gpu")

y = lucid.einops.rearrange(x, "i j (w h) k -> (i j k) w h", w=2, h=2)
y.backward()

print(y.shape)
print(x.grad.shape)
