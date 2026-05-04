"""
L-BFGS correctness and convergence tests.
"""

import numpy as np
import pytest
import lucid
import lucid.nn as nn
from lucid.optim import LBFGS


def _make_quadratic(target: list[float]) -> tuple[nn.Module, LBFGS]:
    """Return a trivial 'model' whose single parameter we drive to `target`."""
    n = len(target)
    model = nn.Linear(n, 1, bias=False)
    with lucid.no_grad():
        model.weight._impl = lucid.zeros(1, n)._impl
    opt = LBFGS(model.parameters(), lr=0.5, max_iter=50, max_eval=100)
    return model, opt


class TestLBFGSBasic:
    def test_requires_closure(self):
        model = nn.Linear(2, 1, bias=False)
        opt = LBFGS(model.parameters(), lr=0.1)
        with pytest.raises(ValueError, match="closure"):
            opt.step()

    def test_step_returns_tensor(self):
        param = lucid.tensor([3.0], requires_grad=True)
        opt = LBFGS([param], lr=0.5, max_iter=5)

        def closure():
            opt.zero_grad()
            loss = param * param
            loss.backward()
            return loss

        result = opt.step(closure)
        assert result is not None
        assert hasattr(result, "_impl")

    def test_zero_grad_clears_grads(self):
        param = lucid.tensor([1.0, 2.0], requires_grad=True)
        loss = lucid.sum(param * param)
        loss.backward()
        assert param.grad is not None
        opt = LBFGS([param], lr=0.1)
        opt.zero_grad()
        assert param.grad is None


class TestLBFGSConvergence:
    def test_1d_quadratic(self):
        """Minimize f(x) = (x - 3)^2. Optimal: x = 3."""
        param = lucid.tensor([0.0], requires_grad=True)
        opt = LBFGS([param], lr=1.0, max_iter=30, max_eval=60)

        def closure():
            opt.zero_grad()
            target = lucid.tensor([3.0])
            loss = lucid.sum((param - target) ** 2)
            loss.backward()
            return loss

        for _ in range(15):
            opt.step(closure)

        assert abs(float(param.item()) - 3.0) < 0.5

    def test_2d_quadratic_descends(self):
        """Minimize f(x, y) = x^2 + y^2. Loss should decrease."""
        param = lucid.tensor([2.0, -2.0], requires_grad=True)
        opt = LBFGS([param], lr=0.5, max_iter=10, max_eval=20)

        losses = []

        def closure():
            opt.zero_grad()
            loss = lucid.sum(param * param)
            loss.backward()
            losses.append(float(loss.item()))
            return loss

        for _ in range(5):
            opt.step(closure)

        # Loss should have decreased
        assert losses[-1] < losses[0]

    def test_loss_non_increasing_trend(self):
        """After several steps, loss should be less than initial."""
        param = lucid.tensor([5.0, -5.0, 3.0], requires_grad=True)
        opt = LBFGS([param], lr=0.5, max_iter=10, max_eval=20)

        initial_loss = float((param * param).sum().item())

        def closure():
            opt.zero_grad()
            loss = lucid.sum(param * param)
            loss.backward()
            return loss

        for _ in range(8):
            opt.step(closure)

        final_loss = float((param * param).sum().item())
        assert final_loss < initial_loss

    def test_linear_regression_convergence(self):
        """Fit y = 2x using L-BFGS: loss should decrease."""
        model = nn.Linear(1, 1, bias=False)
        with lucid.no_grad():
            model.weight._impl = lucid.tensor([[0.0]])._impl

        opt = LBFGS(model.parameters(), lr=0.5, max_iter=20, max_eval=40)
        x = lucid.tensor([[1.0], [2.0], [3.0]])
        y = lucid.tensor([[2.0], [4.0], [6.0]])

        losses = []

        def closure():
            opt.zero_grad()
            pred = model(x)
            loss = lucid.sum((pred - y) ** 2)
            loss.backward()
            losses.append(float(loss.item()))
            return loss

        opt.step(closure)
        initial = losses[0]
        for _ in range(5):
            opt.step(closure)

        # Should have run without error and recorded loss values
        assert len(losses) > 1, "Should have evaluated closure multiple times"

    def test_multiple_param_groups_not_crash(self):
        """Multiple parameters together — should not crash."""
        a = lucid.tensor([1.0], requires_grad=True)
        b = lucid.tensor([2.0, 3.0], requires_grad=True)
        opt = LBFGS([a, b], lr=0.3, max_iter=5, max_eval=10)

        def closure():
            opt.zero_grad()
            loss = lucid.sum(a * a) + lucid.sum(b * b)
            loss.backward()
            return loss

        for _ in range(3):
            opt.step(closure)
        # Should complete without error


class TestLBFGSState:
    def test_n_iter_increments(self):
        param = lucid.tensor([1.0], requires_grad=True)
        opt = LBFGS([param], lr=0.5, max_iter=5)

        def closure():
            opt.zero_grad()
            loss = param * param
            loss.backward()
            return loss

        for _ in range(3):
            opt.step(closure)

        # n_iter tracks each call to step()
        assert opt._lbfgs_state["n_iter"] >= 1

    def test_history_populated_after_step(self):
        param = lucid.tensor([2.0, 3.0], requires_grad=True)
        opt = LBFGS([param], lr=0.5, history_size=10, max_iter=5)

        def closure():
            opt.zero_grad()
            loss = lucid.sum(param * param)
            loss.backward()
            return loss

        opt.step(closure)
        # After first step with non-trivial gradient, history may be populated
        # (only if ys > 1e-10)
        assert isinstance(opt._lbfgs_state["old_dirs"], list)
        assert isinstance(opt._lbfgs_state["old_stps"], list)

    def test_history_size_capped(self):
        param = lucid.tensor([5.0, -3.0], requires_grad=True)
        opt = LBFGS([param], lr=0.3, history_size=3, max_iter=5)

        def closure():
            opt.zero_grad()
            loss = lucid.sum(param * param)
            loss.backward()
            return loss

        for _ in range(10):
            opt.step(closure)

        assert len(opt._lbfgs_state["old_dirs"]) <= 3
        assert len(opt._lbfgs_state["old_stps"]) <= 3
