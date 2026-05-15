"""Tiny self-attention block trains end-to-end."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


@pytest.mark.slow
class TestTransformerTraining:
    def test_loss_shrinks(self, device: str) -> None:
        rng = np.random.default_rng(0)
        # (B=4, T=6, D=8) → next-token regression target (B, T, D).
        x = rng.uniform(-1.0, 1.0, size=(4, 6, 8)).astype(np.float32)
        y = rng.uniform(-1.0, 1.0, size=(4, 6, 8)).astype(np.float32)
        x_t = lucid.tensor(x, device=device)
        y_t = lucid.tensor(y, device=device)

        block = nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16, batch_first=True
        ).to(device=device)
        head = nn.Linear(8, 8).to(device=device)
        opt = optim.Adam(list(block.parameters()) + list(head.parameters()), lr=0.01)

        first = float(F.mse_loss(head(block(x_t)), y_t).item())
        for _ in range(50):
            opt.zero_grad()
            loss = F.mse_loss(head(block(x_t)), y_t)
            loss.backward()
            opt.step()
        last = float(loss.item())
        assert last < 0.7 * first
