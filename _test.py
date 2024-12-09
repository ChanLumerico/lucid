import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)


x = lucid.random.randn(1, 3, 28, 28, requires_grad=True)

conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding="same")
conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding="same")

relu1 = nn.ReLU()
relu2 = nn.ReLU()

bn1 = nn.BatchNorm2d(6)
bn2 = nn.BatchNorm2d(12)

pool = nn.MaxPool2d(2, 2)


def conv_test(x: lucid.Tensor) -> lucid.Tensor:
    x = relu1(bn1(conv1(x)))
    x = relu2(bn2(conv2(x)))
    x = pool(x)
    x = nn.Dropout(p=0.2)(x)  # tmp
    return x


out = conv_test(x)
out.backward()

print(out[0, 0, :3, :3])

print(out.shape)
print(x.grad.shape)
