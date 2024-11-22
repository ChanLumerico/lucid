import lucid

lucid.random.seed(42)


A = lucid.random.randn(2, 2, requires_grad=True)

U, S, VT = lucid.linalg.svd(A)

U.backward()
S.backward()
VT.backward()

print(A.grad)
