import lucid


x = lucid.tensor([1.0, 3.0], requires_grad=True)
y = lucid.tensor([2.0, 4.0], requires_grad=True)

z = x.T.dot(y) ** 2
z = z @ x

z.backward()

print(lucid.diag([3, 2, 1]))

z.backward()

print(f"z: {z}")
print(f"dz/dx: {x.grad}")
print(f"dz/dy: {y.grad}")
