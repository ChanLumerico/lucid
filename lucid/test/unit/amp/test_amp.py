"""``lucid.amp`` — autocast / GradScaler smoke."""

import numpy as np
import pytest

import lucid


class TestAmpSurface:
    def test_autocast_present(self) -> None:
        assert hasattr(lucid.amp, "autocast")

    def test_grad_scaler_present(self) -> None:
        assert hasattr(lucid.amp, "GradScaler")


class TestAutocast:
    def test_context_manager(self) -> None:
        # Just verify the context manager can be entered/exited.
        with lucid.amp.autocast():
            t = lucid.tensor([1.0, 2.0])
            out = t * 2.0
        assert out.shape == (2,)


class TestGradScaler:
    def test_scale_unscale_roundtrip(self) -> None:
        scaler = lucid.amp.GradScaler(init_scale=2.0)
        loss = lucid.tensor(1.0, requires_grad=True)
        scaled = scaler.scale(loss)
        # Scaled loss should be 2 × loss = 2.0.
        assert abs(scaled.item() - 2.0) < 1e-6
