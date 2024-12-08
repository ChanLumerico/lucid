import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)


x = lucid.random.randn(1, 3, 28, 28, requires_grad=True)

conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding="same")
conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding="same")

relu1 = nn.ReLU()
relu2 = nn.ReLU()

pool = nn.MaxPool2d(2, 2)


def conv_test(x: lucid.Tensor) -> lucid.Tensor:
    x = conv1(x)
    x = conv2(relu1(x))
    x = pool(relu2(x))
    x = F.softmax(x, axis=-1)
    return x


out = conv_test(x)
out.backward()

print(out.shape)
print(x.grad.shape)
