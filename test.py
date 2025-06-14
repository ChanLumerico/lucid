import lucid
import lucid.nn as nn
import lucid.optim as optim
import lucid.models as models
import lucid.transforms as transforms

from lucid.data import DataLoader
from lucid.datasets import MNIST

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

lucid.random.seed(42)


transform = transforms.Compose([transforms.Resize((32, 32))])

train_set = MNIST(
    root="./.data/mnist/train", train=True, download=False, transform=transform
)
test_set = MNIST(
    root="./.data/mnist/test", train=False, download=False, transform=transform
)

device = "cpu"
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 5
batch_size = 100

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size)

model: nn.Module = models.lenet_5(_base_activation=nn.ReLU).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss().to(device)


def normalize(images):
    norm = images / 255.0
    norm = (norm - 0.312) / 0.1745
    return norm


def train(model, train_loader, criterion, optimizer, num_epochs):
    losses = []
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        progress_bar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        for batch_idx, (images, labels) in progress_bar:
            images = normalize(images)
            images = images.reshape(batch_size, 1, 32, 32)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels).eval()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            losses.append(loss.item())

            correct += (labels.data == outputs.data.argmax(axis=1)).sum()
            avg_loss = total_loss / batch_idx
            accuracy = correct / (batch_idx * batch_size)

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "AvgLoss": f"{avg_loss:.4f}",
                    "Acc": f"{accuracy:.4f}",
                }
            )

    return losses


losses = train(model, train_loader, criterion, optimizer, num_epochs)

plt.plot(losses, lw=1, label=f"{model._alt_name}")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("MNIST Training Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()
