"""
Tests for optimizers and LR schedulers, especially state preservation.
"""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
import lucid.optim as optim
from conftest import assert_close


class TestOptimizerStatePreservation:
    """Verify optimizer state (Adam m/v, SGD momentum) survives LR scheduler steps."""

    def _run_steps(self, model, opt, n=5):
        for _ in range(n):
            x = lucid.randn(4, 10)
            loss = model(x).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def test_adam_state_preserved_after_steplr(self):
        model = nn.Linear(10, 1)
        opt = optim.Adam(list(model.parameters()), lr=1e-3)
        sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

        eng_id_before = id(opt._engine_optims[0])
        self._run_steps(model, opt)

        sched.step()  # epoch 0 — no decay
        sched.step()  # epoch 1 — decay by gamma

        assert id(opt._engine_optims[0]) == eng_id_before, "Engine optimizer was recreated!"
        expected_lr = 1e-3 * 0.5
        assert abs(opt.param_groups[0]["lr"] - expected_lr) < 1e-10
        assert abs(opt._engine_optims[0].lr - expected_lr) < 1e-10

    def test_sgd_state_preserved(self):
        model = nn.Linear(10, 1)
        opt = optim.SGD(list(model.parameters()), lr=0.1)
        eng_id = id(opt._engine_optims[0])
        self._run_steps(model, opt)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        sched.step()
        assert id(opt._engine_optims[0]) == eng_id


class TestLRSchedulers:
    def _make_model_and_opt(self):
        model = nn.Linear(2, 1)
        opt = optim.Adam(list(model.parameters()), lr=1e-2)
        return model, opt

    def test_steplr(self):
        _, opt = self._make_model_and_opt()
        sched = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
        sched.step()  # epoch 0 — no change
        assert abs(opt.param_groups[0]["lr"] - 1e-2) < 1e-10
        sched.step()  # epoch 1 — no change (step_size=2)
        assert abs(opt.param_groups[0]["lr"] - 1e-2) < 1e-10
        sched.step()  # epoch 2 — decay!
        assert abs(opt.param_groups[0]["lr"] - 5e-3) < 1e-10

    def test_cosine_annealing(self):
        _, opt = self._make_model_and_opt()
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0)
        # After T_max steps, LR should reach eta_min
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] <= 1e-2

    def test_reduce_on_plateau(self):
        _, opt = self._make_model_and_opt()
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
        for _ in range(4):
            sched.step(1.0)  # no improvement 4 times
        assert abs(opt.param_groups[0]["lr"] - 5e-3) < 1e-10


class TestClipGradNorm:
    def test_clip_preserves_direction(self):
        fc = nn.Linear(4, 2)
        x = lucid.randn(3, 4)
        fc(x).mean().backward()

        import lucid.nn.utils as nnu
        import numpy as np

        # Save grad direction
        g_before = fc.weight.grad.numpy().copy()
        norm_before = np.linalg.norm(g_before)

        nnu.clip_grad_norm_(list(fc.parameters()), max_norm=0.1)

        g_after = fc.weight.grad.numpy()
        norm_after = np.linalg.norm(g_after)

        assert norm_after <= 0.1 + 1e-5
        if norm_before > 1e-8:
            # Direction preserved: cosine similarity ≈ 1
            cos = np.dot(g_before.flat, g_after.flat) / (norm_before * norm_after + 1e-12)
            assert cos > 0.99
