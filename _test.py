import lucid

lucid.random.seed(42)


A = lucid.random.randn(2, 2, requires_grad=True)
B = lucid.random.randn(2, 2, requires_grad=True)

C = lucid.linalg.pinv(A)
C.backward()

print(A.grad)
