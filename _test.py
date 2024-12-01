import lucid
import lucid.nn.functional as F


input_ = lucid.random.randn(1, 1, 3, 3, requires_grad=True)
weight = lucid.random.randn(1, 1, 2, 2, requires_grad=True)
bias = lucid.zeros((1,), requires_grad=True)

out = F.unfold(input_, (2, 2), (1, 1), (0, 0))

print(f"out-shape: {out.shape}")

out.backward()
print(input_.grad)