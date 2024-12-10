import lucid
import lucid.nn as nn
import lucid.nn.functional as F

lucid.random.seed(42)


x = lucid.random.randn(10, 3, 28, 28, requires_grad=True)
y = lucid.random.rand(10) >= 0.5

conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1)
conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1)

relu1 = nn.ReLU()
relu2 = nn.ReLU()

bn1 = nn.BatchNorm2d(6)
bn2 = nn.BatchNorm2d(12)

pool = nn.MaxPool2d(2, 2)
drop = nn.Dropout(p=0.2)

fc1 = nn.Linear(1728, 100)
fc2 = nn.Linear(100, 10)

soft = nn.Softmax()
loss = nn.CrossEntropyLoss()


def conv_test(x: lucid.Tensor) -> lucid.Tensor:
    x = relu1(bn1(conv1(x)))
    x = relu2(bn2(conv2(x)))
    x = pool(x)
    x = drop(x)

    x = x.reshape(x.shape[0], -1)
    x = fc2(fc1(x))
    x = soft(x)

    return loss(x, target=y)


out = conv_test(x)
out.backward()

print(out.shape if out.ndim > 0 else out.item())
print(x.grad.shape)
