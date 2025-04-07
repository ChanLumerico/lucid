import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import lucid.models as models

import numpy as np
import mlx.core as mx

import time

lucid.random.seed(42)

device = "gpu"


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.avgpool(x)
        x = x.flatten(axis=1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


module = MyModule()
module.to(device)

optimizer = optim.RMSprop(module.parameters())
criterion = nn.MSELoss()


x = lucid.random.randn(10, 3, 64, 64).to(device)
y = lucid.random.randn(10, 1).to(device)

t0 = time.time_ns()

for i in range(5):
    out = module(x)
    # out.backward()

    loss = criterion(out, y).eval()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"[{i + 1}] {loss.item()}")

t1 = time.time_ns()

print(out.device, out.shape)
print(f"{(t1 - t0) / 1e6} ms")
