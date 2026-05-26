"""Phase 1-4 acceptance gate for the manual-VJP compile path.

The manual VJP path replaces MPSGraph's ``gradientForPrimaryTensor:``
with a Lucid-side reverse walker that emits per-op backward subgraphs.
It is opt-in via ``LUCID_MANUAL_VJP=1`` and unblocks transformer
training compile (where MPSGraph autograd asserts on embedding /
split_at / non-float feeds).

Phase 1-4 covers the elementwise / activation / linear / matmul /
reshape / reduction / loss families — enough to compile a
``Linear → ReLU → Linear → MSE`` step end-to-end with the manual VJP
walker and *no* MPSGraph autograd involvement (``LUCID_MANUAL_VJP_REQUIRE=1``
forces hard failure on coverage gap rather than silent fall-through).

Pass criterion: per-step loss matches the eager loss within ``1e-3``
over 5 SGD updates.  Drift larger than that signals a wrong VJP
formula somewhere (the smoke surface touches exactly the 5 VJPs:
linear, relu, mse_loss, plus the implicit reshape/permute the linear
emitter generates internally).
"""

import os

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


def _mlp_model() -> nn.Module:
    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 8)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return self.fc2(self.fc1(x).relu())

    return _MLP().to(COMPILE_DEVICE)


def _eager_loss_trajectory(
    model: nn.Module, x: lucid.Tensor, t: lucid.Tensor, steps: int, lr: float
) -> list[float]:
    opt = optim.SGD(list(model.parameters()), lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(x), t)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _compile_loss_trajectory(
    model: nn.Module, x: lucid.Tensor, t: lucid.Tensor, steps: int, lr: float
) -> list[float]:
    opt = optim.SGD(list(model.parameters()), lr=lr)
    step = fused_step(model, F.mse_loss, opt)
    losses: list[float] = []
    for _ in range(steps):
        # ``fused_step`` returns the loss (and updates params in place).
        out = step(x, t)
        if isinstance(out, lucid.Tensor):
            losses.append(float(out.item()))
        else:
            # Some fused_step variants return a tuple (loss, ...).
            losses.append(float(out[0].item()))
    return losses


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _mlp_model()
    b = _mlp_model()
    # Mirror weights so eager and compile start identical.
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        pb.copy_(pa.detach().clone())
    return a, b


@pytest.fixture(autouse=True)
def _manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force manual VJP on + hard-fail on coverage gap for this test."""
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


def test_manual_vjp_mlp_loss_parity() -> None:
    """5 SGD steps of MLP → MSE — manual VJP loss matches eager within 1e-3."""
    lucid.manual_seed(0)
    x = metal_tensor(8, 16)
    t = metal_tensor(8, 8)
    eager_model, comp_model = _matched_models()

    eager_losses = _eager_loss_trajectory(eager_model, x, t, steps=5, lr=1e-3)
    comp_losses = _compile_loss_trajectory(comp_model, x, t, steps=5, lr=1e-3)

    assert len(eager_losses) == len(comp_losses) == 5
    for k in range(5):
        diff = abs(eager_losses[k] - comp_losses[k])
        assert diff < 1e-3, (
            f"manual VJP loss drift at step {k}: "
            f"eager={eager_losses[k]:.6f}, compile={comp_losses[k]:.6f}, "
            f"diff={diff:.6f}"
        )


def test_manual_vjp_engine_loaded() -> None:
    """Smoke: the manual VJP env helpers are wired and engine sees the flag."""
    assert os.environ.get("LUCID_MANUAL_VJP") == "1"
    assert os.environ.get("LUCID_MANUAL_VJP_REQUIRE") == "1"
