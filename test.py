import lucid
import lucid.nn as nn
import lucid.optim as optim
import lucid.models as models
import lucid.transforms as transforms

from lucid.data import DataLoader
from lucid.datasets import MNIST

import numpy as np


transform = transforms.Compose([transforms.Resize((48, 48))])

train_set = MNIST(
    root="./.data/mnist/train", train=True, download=True, transform=transform
)
test_set = MNIST(
    root="./.data/mnist/test", train=False, download=True, transform=transform
)

learning_rate = 1e-4
num_epochs = 3
batch_size = 16
weight_decay = 0.001

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size)

vit_tiny: nn.Module = models.vit_tiny(image_size=48, num_classes=10)

optimizer = optim.Adam(
    vit_tiny.parameters(), lr=learning_rate, weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = lucid.repeat(images, 3, axis=1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (labels.data == outputs.data.argmax(axis=1)).sum()

            print(
                f"Batch [{batch_idx}/{len(train_loader)}] -",
                f"Loss: {loss.item():.4f},",
                f"Avg. Loss: {total_loss / batch_idx:.4f}",
                f"Accuracy: {correct / (batch_idx * batch_size):.4f}",
            )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}\n")


train(vit_tiny, train_loader, criterion, optimizer, num_epochs)
