import lucid
import lucid.nn.functional as F

lucid.random.seed(42)


a = lucid.tensor([1, 2], requires_grad=True)

b = F.gelu(a)
b.backward()

print(a.grad)
