import lucid
import lucid.nn.functional as F

lucid.random.seed(42)


X1 = lucid.random.randn(10, 3, requires_grad=True)
X2 = lucid.random.randn(10, 4, requires_grad=True)

W = lucid.random.randn(6, 3, 4, requires_grad=True) * 0.1
b = lucid.zeros((1, 6), requires_grad=True)

out = F.bilinear(X1, X2, W, b)

norm = lucid.linalg.norm(out, ord=2)
norm.backward()

print(X1.grad)
print(X2.grad)
