import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_sc = sc.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.3, random_state=42
)

X_train = lucid.Tensor(X_train)
y_train = lucid.Tensor(y_train)

batch_size = 100
num_epochs = 10
lr = 0.01

import numpy as np

num_samples = X_train.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

model = MLP(input_size=784, hidden_size=100, output_size=10)
optimizer = optim.SGD(model.parameters(), lr=lr)


def train(epoch):
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = list(indices[start_idx:end_idx])

        X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

        out = model(X_batch)
        loss = F.cross_entropy(out, y_batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if start_idx % 1000 == 0:
            loss_history.append(loss.item())

        if start_idx % 5000 == 0 or start_idx == 0:
            print(f"Epoch {epoch} - Batch {start_idx}, Loss: {loss.item()}")


loss_history = []
for epoch in range(1, num_epochs + 1):
    train(epoch)
    print()


import matplotlib.pyplot as plt

plt.plot(loss_history, lw=1)
plt.tight_layout()
plt.savefig("mnist_mlp_test")
