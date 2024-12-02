import lucid
import lucid.nn.functional as F

lucid.random.seed(42)


X = lucid.random.randn(10, 32, 28, 28, requires_grad=True)

W1 = lucid.random.randn(64, 32, 3, 3, requires_grad=True)
W2 = lucid.random.randn(128, 64, 3, 3, requires_grad=True)

X = F.conv2d(X, W1, stride=1, padding=0)
X = F.conv2d(F.relu(X), W2, stride=1, padding=0)
X = F.avg_pool2d(X, kernel_size=2, stride=2, padding=0)
X = F.max_pool2d(X, kernel_size=2, stride=2, padding=0)

print(X.shape)

X.backward()

print(W1.grad.shape)
print(W2.grad.shape)
