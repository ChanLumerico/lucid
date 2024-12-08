import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)


x = lucid.random.randn(1, 3, 28, 28, requires_grad=True)

conv1 = nn.Conv2D(3, 6, kernel_size=3, stride=1, padding=0)
conv2 = nn.Conv2D(6, 12, kernel_size=3, stride=1, padding=0)


def conv_test(x: lucid.Tensor) -> lucid.Tensor:
    x = conv1(x)
    x = conv2(F.relu(x))
    return F.relu(x)


out = conv_test(x)
out.backward()

print(out.shape)
print(x.grad.shape)
