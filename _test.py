import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)


x = lucid.random.randn(1, 3, 28, 28, requires_grad=True)
w = lucid.random.randn(6, 1, 3, 3, requires_grad=True) * 0.01

out = F.conv2d(x, w, stride=1, padding=0, dilation=1, groups=3)
out.backward()

print(out.shape)
print(x.grad.shape)
print(w.grad.shape)
