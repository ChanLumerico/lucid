import lucid


x = lucid.Tensor([1, 2, 3], requires_grad=True)

# x.to("gpu")

y = x + x
y.backward()

print(x.grad)
