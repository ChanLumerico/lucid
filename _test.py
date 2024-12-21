import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim

from lucid import datasets
from lucid.data import DataLoader


mnist_train = datasets.MNIST(root="./_data/mnist", train=True)
mnist_test = datasets.MNIST(root="./_data/mnist", train=False)

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)


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


def standard_scale(image):
    mean = image.mean(axis=(2, 3), keepdims=True)
    std = image.var(axis=(2, 3), keepdims=True) ** 0.5

    return (image - mean) / std


def train(model, optimizer, epoch):
    loss_arr = []
    for i, (X_batch, y_batch) in enumerate(train_loader, start=1):
        X_batch = standard_scale(X_batch)

        y_pred = model(X_batch)
        loss = F.cross_entropy(y_pred, y_batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_arr.append(loss.item())

        if i % 50 == 0:
            print(f"[{type(optimizer).__name__}]", end=" ")
            print(f"Epoch {epoch} - Batch {i}, Loss {loss.item()}")

    return loss_arr


def fit_model(model, optimizer):
    loss_hist = []
    for epoch in range(1, num_epochs + 1):
        loss_arr = train(model, optimizer, epoch)
        loss_avg = lucid.mean(loss_arr).item()
        loss_hist.extend(loss_arr)

        print(f"[{type(optimizer).__name__}]", end=" ")
        print(f"Epoch {epoch}/{num_epochs}, Avg. Loss {loss_avg}\n")

    return loss_hist


# =======================[ Train & Eval ]======================== #

import matplotlib.pyplot as plt

optimizers_list = [
    optim.SGD,
    optim.RMSprop,
    optim.Adam,
    # optim.AdamW,
    # optim.NAdam,
    # optim.Adamax,
    # optim.Adagrad,
    # optim.Adadelta,
]

model_type = LeNet5

batch_size = 64
num_epochs = 1
lr = 0.001

plt.figure(figsize=(10, 5))

for optimizer in optimizers_list:
    model_ = model_type()
    optim_ = optimizer(
        model_.parameters(), lr=lr if optimizer is not optim.Adadelta else 1.0
    )

    loss_arr_ = fit_model(model=model_, optimizer=optim_)
    plt.plot(loss_arr_, label=optimizer.__name__, lw=0.5, alpha=0.7)

plt.xlabel("Batches")
plt.ylabel("Cross-Entropy Loss")
plt.title(f"LeNet-5 on Lucid for MNIST")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig(f"lenet5_optims_comp")
