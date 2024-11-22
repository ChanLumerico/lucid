import lucid

lucid.random.seed(42)


A = lucid.random.randn(2, 2, requires_grad=True)
B = lucid.random.randn(2, 2, requires_grad=True)

C = lucid.linalg.matrix_power(A, 2)
C.backward()

print(A.grad)
