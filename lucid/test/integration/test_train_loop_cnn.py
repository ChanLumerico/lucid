"""End-to-end CNN training on a tiny synthetic image-classification
problem.  Same loss-shrinks contract as the MLP test."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


@pytest.mark.slow
class TestCNNTraining:
    # Conv2d backward + reshape on Metal trips a shape mismatch in the
    # current MLX adapter; pin to CPU until the engine fix lands.
    def test_loss_shrinks(self, device_cpu_only: str) -> None:
        device = device_cpu_only
        rng = np.random.default_rng(0)
        # 16 grayscale 8×8 "images", 3 classes.
        x = rng.uniform(-1.0, 1.0, size=(16, 1, 8, 8)).astype(np.float32)
        y = rng.integers(0, 3, size=(16,)).astype(np.int64)

        x_t = lucid.tensor(x, device=device)
        y_t = lucid.tensor(y, dtype=lucid.int64, device=device)

        model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (4, 4, 4)
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 3),
        ).to(device=device)

        opt = optim.Adam(model.parameters(), lr=0.05)

        first = float(F.cross_entropy(model(x_t), y_t).item())
        for _ in range(60):
            opt.zero_grad()
            loss = F.cross_entropy(model(x_t), y_t)
            loss.backward()
            opt.step()
        last = float(loss.item())
        assert last < 0.5 * first
