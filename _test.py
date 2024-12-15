import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train).reshape(-1, 1, 28, 28)

input_ = lucid.Tensor(X_train_sc)
target = lucid.Tensor(y_train)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.reshape(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.NAdam(model.parameters(), lr=0.002)

batch_size = 64
num_epochs = 10
num_samples = input_.shape[0]

indices = lucid.arange(num_samples).astype(int)


def train(model, optimizer, epoch):
    loss_arr = []
    for i, start_idx in enumerate(range(0, num_samples, batch_size), start=1):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        X_batch, y_batch = input_[batch_indices], target[batch_indices]

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_arr.append(loss.item())

        if i % 10 == 0:
            print(f"Epoch {epoch} - Batch {i}, Loss {loss.item()}")

    return loss_arr


def fit_model(model, optimizer):
    loss_hist = []
    for epoch in range(1, num_epochs + 1):
        loss_arr = train(model, optimizer, epoch)
        loss_avg = lucid.mean(loss_arr).item()
        loss_hist.extend(loss_arr)

        print(f"Epoch {epoch}/{num_epochs}, Avg. Loss {loss_avg}\n")

    with lucid.no_grad():
        import numpy as np

        y_test_pred = model(lucid.to_tensor(X_test).reshape(-1, 1, 28, 28))
        y_test_soft = F.softmax(y_test_pred)
        y_test_out = np.argmax(y_test_soft.data, axis=1)

        accuracy = np.sum(y_test == y_test_out) / y_test.size

    return loss_hist, accuracy


loss_arr, acc = fit_model(model, optimizer)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(loss_arr, label=type(optimizer).__name__, lw=0.75, c="orange")

plt.xlabel("Entire Batches")
plt.ylabel("Cross-Entropy Loss")
plt.title(f"LeNet-5 on Lucid for MNIST [Acc: {acc:.4f}]")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()

plt.savefig(f"lenet5_mnist_{type(optimizer).__name__.lower()}")
