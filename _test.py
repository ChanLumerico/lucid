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
X_train_sc = sc.fit_transform(X_train)

input_ = lucid.Tensor(X_train_sc)
target = lucid.Tensor(y_train)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model_sgd = MLP()
model_rms = MLP()

sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
rmsprop = optim.Rprop(model_rms.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

batch_size = 64
num_epochs = 30
num_samples = input_.shape[0]

indices = lucid.arange(num_samples).astype(int)


def train(model, optimizer):
    loss_arr = []
    for i, start_idx in enumerate(range(0, num_samples, batch_size)):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        X_batch, y_batch = input_[batch_indices], target[batch_indices]

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_arr.append(loss.item())

    return lucid.mean(loss_arr).item()


def fit_model(model, optimizer):
    loss_hist = []
    for epoch in range(1, num_epochs + 1):
        loss_avg = train(model, optimizer)
        loss_hist.append(loss_avg)

        print(f"Epoch {epoch}/{num_epochs}, Avg. Loss {loss_avg}")

    with lucid.no_grad():
        import numpy as np

        y_test_pred = model(lucid.to_tensor(X_test))
        y_test_soft = F.softmax(y_test_pred)
        y_test_out = np.argmax(y_test_soft.data, axis=1)

        accuracy = np.sum(y_test == y_test_out) / y_test.size

    return loss_hist, accuracy


loss_rms, acc_rms = fit_model(model_rms, rmsprop)
loss_sgd, acc_sgd = fit_model(model_sgd, sgd)

best_acc = max(acc_sgd, acc_rms)

import matplotlib.pyplot as plt

plt.plot(loss_sgd, label="SGD")
plt.plot(loss_rms, label="RMSprop")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title(f"Lucid Test on MLP for MNIST [Acc: {best_acc:.4f}]")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()

plt.savefig(f"mlp_test_sgd_rmsprop")
