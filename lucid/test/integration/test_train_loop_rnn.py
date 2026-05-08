"""End-to-end RNN / LSTM training.  Tiny sequence task: regress a
target from a 5-step input."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


@pytest.mark.slow
class TestLSTMTraining:
    # LSTM backward currently exercises a Metal codepath that diverges
    # in the engine adapter — pin to CPU.
    def test_loss_shrinks(self, device_cpu_only: str) -> None:
        device = device_cpu_only
        rng = np.random.default_rng(0)
        # (T=5, B=8, F=3) → (B, 1) regression target.
        x = rng.uniform(-1.0, 1.0, size=(5, 8, 3)).astype(np.float32)
        y = rng.uniform(-1.0, 1.0, size=(8, 1)).astype(np.float32)

        x_t = lucid.tensor(x, device=device)
        y_t = lucid.tensor(y, device=device)

        rnn = nn.LSTM(input_size=3, hidden_size=8, num_layers=1).to(device=device)
        head = nn.Linear(8, 1).to(device=device)

        opt = optim.Adam(list(rnn.parameters()) + list(head.parameters()), lr=0.05)

        def forward() -> lucid.Tensor:
            out, _ = rnn(x_t)
            # last time-step.
            return head(out[-1])

        first = float(F.mse_loss(forward(), y_t).item())
        for _ in range(40):
            opt.zero_grad()
            loss = F.mse_loss(forward(), y_t)
            loss.backward()
            opt.step()
        last = float(loss.item())
        assert last < 0.5 * first
