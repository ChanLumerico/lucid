import lucid

lucid.random.seed(42)


A = lucid.random.randn(2, 2, requires_grad=True)

B = lucid.linalg.pinv(A)
B.backward()

print(A.grad)
