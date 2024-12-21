import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim

from lucid import models
from lucid import datasets, transforms
from lucid.data import DataLoader

mnist_train = datasets.MNIST(root="./_data/mnist", train=True)
mnist_test = datasets.MNIST(root="./_data/mnist", train=False)

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)


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

batch_size = 64
num_epochs = 1
lr = 0.001

plt.figure(figsize=(10, 5))

for optimizer in optimizers_list:
    model_ = models.lenet_5()
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
