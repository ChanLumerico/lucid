import lucid

lucid.random.seed(42)


A = lucid.random.randn(2, 2, requires_grad=True)

L, V = lucid.linalg.eig(A)
L.backward()
V.backward()

print(A.grad)
