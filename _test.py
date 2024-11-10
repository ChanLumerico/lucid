import lucid


x = lucid.tensor([1, 2], requires_grad=True)
y = lucid.tensor([3, 4], requires_grad=True)

z = x.dot(y)

z.backward(keep_grad=True)

print(z)

print(z.grad)
print(x.grad)
print(y.grad)