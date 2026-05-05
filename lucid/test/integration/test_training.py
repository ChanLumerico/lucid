"""
End-to-end training loop tests.  These are marked @slow and run in ci_full.sh.
"""

import pytest
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.test.helpers.numerics import make_tensor


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TestSGDTraining:
    @pytest.mark.slow
    def test_loss_decreases_sgd(self):
        lucid.manual_seed(0)
        model = TinyMLP()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        x = make_tensor((16, 8))
        target = make_tensor((16, 4), seed=1)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        # Loss should strictly decrease after a few steps
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    @pytest.mark.slow
    def test_adam_converges(self):
        lucid.manual_seed(42)
        model = TinyMLP()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        x = make_tensor((16, 8))
        target = make_tensor((16, 4), seed=1)

        initial_loss = None
        for step in range(10):
            opt.zero_grad()
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            opt.step()
            if initial_loss is None:
                initial_loss = float(loss.item())
        final_loss = float(loss.item())
        assert (
            final_loss < initial_loss
        ), f"Adam did not converge: {initial_loss} → {final_loss}"


class TestGradientFlow:
    @pytest.mark.slow
    def test_gradients_not_none_after_backward(self):
        model = TinyMLP()
        x = make_tensor((4, 8))
        out = model(x)
        lucid.sum(out).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"

    @pytest.mark.slow
    def test_zero_grad_clears_gradients(self):
        model = TinyMLP()
        opt = optim.SGD(model.parameters(), lr=0.01)
        x = make_tensor((4, 8))
        out = model(x)
        lucid.sum(out).backward()
        opt.zero_grad()
        for p in model.parameters():
            assert p.grad is None
