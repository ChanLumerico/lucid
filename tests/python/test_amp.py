"""
Tests for lucid.amp — autocast and GradScaler.
"""

import pytest
import lucid
import lucid.amp as amp
import lucid.nn as nn


class TestAutocast:
    def test_context_manager(self):
        x = lucid.randn(3, 4)
        with amp.autocast():
            y = x * 2.0
        assert y is not None

    def test_decorator(self):
        @amp.autocast()
        def forward(x):
            return x * 2.0

        x = lucid.randn(3)
        y = forward(x)
        assert y is not None

    def test_nested_no_crash(self):
        with amp.autocast():
            with amp.autocast():
                x = lucid.randn(2)
                y = x + 1.0
        assert y is not None

    def test_disabled(self):
        x = lucid.randn(3)
        with amp.autocast(enabled=False):
            y = x * 2.0
        assert y is not None


class TestGradScaler:
    def _simple_model_step(self, scaler: amp.GradScaler) -> tuple[lucid.Tensor, bool]:
        model = nn.Linear(4, 2)
        opt = lucid.optim.SGD(model.parameters(), lr=0.01)
        x = lucid.randn(2, 4)
        with amp.autocast():
            out = model(x)
            loss = lucid.sum(out)
        opt.zero_grad()
        scaler.scale(loss).backward()
        updated = scaler.step(opt)
        scaler.update()
        return loss, updated

    def test_scale_returns_tensor(self):
        scaler = amp.GradScaler(init_scale=128.0)
        loss = lucid.tensor(1.0)
        scaled = scaler.scale(loss)
        assert hasattr(scaled, "_impl")

    def test_step_and_update(self):
        scaler = amp.GradScaler(init_scale=128.0)
        loss, _ = self._simple_model_step(scaler)
        assert float(loss.item()) == float(loss.item())  # not NaN

    def test_get_scale(self):
        scaler = amp.GradScaler(init_scale=256.0)
        assert scaler.get_scale() == 256.0

    def test_state_dict_round_trip(self):
        scaler = amp.GradScaler(init_scale=512.0, growth_interval=100)
        sd = scaler.state_dict()
        scaler2 = amp.GradScaler()
        scaler2.load_state_dict(sd)
        assert scaler2.get_scale() == 512.0

    def test_enabled_false_passthrough(self):
        scaler = amp.GradScaler(enabled=False)
        loss = lucid.tensor(2.0, requires_grad=True)
        scaled = scaler.scale(loss)
        # When disabled, scale should be identity (scale factor = 1)
        assert abs(float(scaled.item()) - 2.0) < 1e-5
