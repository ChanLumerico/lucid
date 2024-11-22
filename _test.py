import lucid

lucid.random.seed(10)


A = lucid.random.randn(3, requires_grad=True)
B = lucid.random.randn(3, requires_grad=True)

C = lucid.outer(A, B)
C.backward()

print(A.grad)
print(B.grad)
