import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)


x = lucid.random.randn(10, 3, 28, 28, requires_grad=True)
w = lucid.random.randn(3, 3, 3, 3, requires_grad=True) * 0.01

out = F.conv2d(x, w, stride=1, padding=0, dilation=2)
out = F.conv2d(out, w, stride=2, padding=1, dilation=1)
out.backward()

print(out.shape)
print(x.grad.shape)
print(w.grad.shape)
