import lucid
import lucid.nn.functional as F


input_ = lucid.random.randn(10, 32, 28, 28, requires_grad=True)
weight = lucid.random.randn(64, 32, 3, 3, requires_grad=True)
bias = lucid.zeros((64,), requires_grad=True)

out = F.conv2d(input_, weight, bias, stride=1, padding=0)
out.backward()

print(out.shape)
print(input_.shape)
print(weight.shape)
print(bias.shape)
