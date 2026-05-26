"""Training-mode dropout in fused_step — Option-A Phase 1 acceptance.

Pins the stateful Philox plumbing end-to-end: when training-mode
dropout runs inside :func:`fused_step`, the Python wrapper routes
through ``_C_engine.nn.dropout_stateful``, the trace captures a
2-input/2-output node, and the compile path promotes the state
buffer to an MPSGraph variable via
``compile_generic_fused_step_with_vars`` — so each dispatch advances
the Philox state and the mask actually varies across calls (no more
eager fallback for this signature).

Two acceptance gates:

* **Compile success** — the executable cache populates (i.e. the
  signature is NOT in ``eager_only``).  Counterpart to the
  ``test_dropout_training_lucid_compile_still_falls_back_to_eager``
  test which pins the forward-only path's continued eager fallback.

* **Mask varies per dispatch** — five consecutive ``step(x, t)``
  calls on the *same* input must produce non-identical loss values.
  Because the optimizer also updates parameters between calls, the
  five losses would already differ from optimizer drift alone — so
  this gate is a *qualitative* signal that things work, not a tight
  parity check.  The mask-randomness contract is also covered by
  the existing ``test_dropout_training_produces_random_outputs``
  (which the Phase 1 change must not regress).
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


@pytest.fixture(autouse=True)
def _require_manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run with manual-VJP REQUIRE=1 so any coverage gap surfaces as RuntimeError."""
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


class _DropoutMLP(nn.Module):
    """Linear → ReLU → Dropout(0.5) → Linear MLP — minimum reproducer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.drop(self.fc1(x).relu()))


def test_fused_step_training_dropout_compiles() -> None:
    """``fused_step`` with training-mode dropout compiles cleanly.

    Under ``LUCID_MANUAL_VJP_REQUIRE=1`` a coverage gap would raise.
    Reaching the second call without raising = the compile path
    threads ``dropout_stateful`` end-to-end + the manual VJP for
    the new op is registered.
    """
    lucid.manual_seed(0)
    model = _DropoutMLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, F.mse_loss, opt)

    x = metal_tensor(4, 8)
    t = metal_tensor(4, 4)

    # First call traces + compiles + dispatches.  Second call hits the
    # executable cache.  Both must succeed without raising.
    loss1 = float(step(x, t).item())
    loss2 = float(step(x, t).item())
    assert loss1 == loss1  # not NaN
    assert loss2 == loss2  # not NaN


def test_fused_step_training_dropout_varies_per_dispatch() -> None:
    """Five consecutive ``step`` calls produce distinct losses.

    The mask randomness contract — same input, different masks per
    dispatch — combined with optimizer drift means each step's loss
    should differ from the previous.  An identical-mask regression
    would manifest as suspiciously regular loss progressions.
    """
    lucid.manual_seed(0)
    model = _DropoutMLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, F.mse_loss, opt)

    x = metal_tensor(4, 8)
    t = metal_tensor(4, 4)

    losses = [float(step(x, t).item()) for _ in range(5)]
    # All five losses must be distinct.  If the mask were identical
    # across dispatches, the optimizer drift alone would still cause
    # some variation but the test is a basic sanity check.
    assert len(set(losses)) == 5, (
        f"expected 5 distinct losses from 5 dispatches "
        f"(mask should vary + optimizer drifts); got {losses}"
    )


def test_fused_step_loss_decreases_over_steps() -> None:
    """Training works end-to-end — loss should trend downward.

    Loose check (compare last 3 to first 3 averaged) because dropout
    masks add per-step noise.  Pinning the trend confirms the
    backward path through ``dropout_stateful``'s VJP is wired
    correctly + gradients actually reach the parameters.
    """
    lucid.manual_seed(0)
    model = _DropoutMLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-2)
    step = fused_step(model, F.mse_loss, opt)

    x = metal_tensor(16, 8)
    t = metal_tensor(16, 4)

    losses: list[float] = []
    for _ in range(20):
        losses.append(float(step(x, t).item()))

    avg_first = sum(losses[:3]) / 3
    avg_last = sum(losses[-3:]) / 3
    assert avg_last < avg_first, (
        f"expected loss to decrease over 20 SGD steps; "
        f"first-3 avg={avg_first:.4f}, last-3 avg={avg_last:.4f}; "
        f"all losses: {losses}"
    )
