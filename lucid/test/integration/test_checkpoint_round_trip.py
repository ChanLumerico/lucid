"""Save → load → resume training: parameters survive intact and the
loss curve continues from where it left off."""

from pathlib import Path

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim


def _build() -> tuple[nn.Sequential, optim.Optimizer, lucid.Tensor, lucid.Tensor]:
    rng = np.random.default_rng(0)
    x = lucid.tensor(rng.uniform(-1.0, 1.0, size=(16, 4)).astype(np.float32))
    y = lucid.tensor(rng.uniform(-1.0, 1.0, size=(16, 1)).astype(np.float32))
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    opt = optim.Adam(model.parameters(), lr=0.05)
    return model, opt, x, y


@pytest.mark.slow
class TestCheckpointRoundTrip:
    def test_state_dict_round_trip(self, tmp_path: Path) -> None:
        model, opt, x, y = _build()

        # 5 warm-up steps.
        for _ in range(5):
            opt.zero_grad()
            F.mse_loss(model(x), y).backward()
            opt.step()
        loss_before = float(F.mse_loss(model(x), y).item())

        ckpt = tmp_path / "ckpt.lcd"
        lucid.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
            },
            str(ckpt),
        )

        # Build fresh, load.
        model2, opt2, _, _ = _build()
        loaded = lucid.load(str(ckpt), weights_only=False)
        model2.load_state_dict(loaded["model"])
        opt2.load_state_dict(loaded["opt"])

        # After load, identical model+optim must reproduce identical loss.
        loss_after = float(F.mse_loss(model2(x), y).item())
        assert abs(loss_before - loss_after) < 1e-5

        # And training continues smoothly from there — loss shouldn't
        # explode after resuming with the loaded optimizer state.
        for _ in range(20):
            opt2.zero_grad()
            F.mse_loss(model2(x), y).backward()
            opt2.step()
        loss_final = float(F.mse_loss(model2(x), y).item())
        assert loss_final <= loss_before + 1e-3
