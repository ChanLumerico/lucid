import lucid
import lucid.nn as nn
import lucid.optim as optim
import lucid.models as models
import lucid.transforms as transforms

from lucid.data import DataLoader
from lucid.datasets import MNIST

import numpy as np


transform = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.Normalize((0.5,), (0.5,))]
)

train_set = MNIST(
    root="./.data/mnist/train", train=True, download=False, transform=transform
)
test_set = MNIST(
    root="./.data/mnist/test", train=False, download=False, transform=transform
)

learning_rate = 1e-3
num_epochs = 3
batch_size = 100
weight_decay = 1e-6

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size)

model: nn.Module = models.lenet_5().to("cpu")

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss().to("cpu")


def normalize(images):
    return (images - images.mean(axis=(2, 3), keepdims=True)) / images.var(
        axis=(2, 3), keepdims=True
    ) ** 0.5


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.reshape(batch_size, 1, 32, 32).to("cpu")
            images = normalize(images)
            labels = labels.to("cpu")

            outputs = model(images)
            loss = criterion(outputs, labels).eval()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (labels.data == outputs.data.argmax(axis=1)).sum()
            avg_loss = total_loss / batch_idx
            accuracy = correct / (batch_idx * batch_size)

            print(
                f"Batch [{batch_idx}/{len(train_loader)}] -",
                f"Loss: {loss.item():.4f},",
                f"Avg. Loss: {avg_loss:.4f}",
                f"Accuracy: {accuracy:.4f}",
            )

            # with open("_vit_t.txt", "w") as f:
            #     f.write(
            #         f"[Epoch {epoch}/{num_epochs} - "
            #         + f"Batch {batch_idx}/{len(train_loader)}]\n"
            #         + f"Loss: {loss.item():.4f}\n"
            #         + f"Avg. Loss: {avg_loss:.4f}\n"
            #         + f"Accuracy: {accuracy:.4f}\n"
            #     )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}\n")


train(model, train_loader, criterion, optimizer, num_epochs)
