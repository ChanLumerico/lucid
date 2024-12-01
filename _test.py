import lucid
import lucid.nn.functional as F


# input_ = lucid.random.randn(10, 32, 28, 28, requires_grad=True)
# weight = lucid.random.randn(64, 32, 3, 3, requires_grad=True)
# bias = lucid.zeros((64,), requires_grad=True)

# out = F.conv2d(input_, weight, bias, stride=1, padding=0)
# out = F.relu(out)
# out = F.avg_pool2d(out, kernel_size=2, stride=2)

# print(f"out-shape: {out.shape}")

# out.backward()

a = lucid.zeros((10, 10), requires_grad=True)
b = lucid.zeros_like(a, requires_grad=True)

c = lucid.stack([a[:3], b[:3]], axis=-1)
c.backward()

print(a.grad.shape)
print(b.grad.shape)
