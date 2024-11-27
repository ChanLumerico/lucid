import lucid
import lucid.nn.functional as F

lucid.random.seed(42)

a = lucid.tensor([[-1, 2], [3, -4]], requires_grad=True)
b = a.tile((2, 3))
b = lucid.linalg.norm(b)

b.backward()

print(b)
print(a.grad)
