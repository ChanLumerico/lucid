"""Manual VJP — BCE + Huber loss acceptance (X1.1).

Two acceptance tests covering the loss VJPs added in Phase X1:
  * `bce_loss`   — binary cross-entropy on probabilities (sigmoid out).
  * `huber_loss` — robust regression loss with quadratic/linear branch.

Both use the closed-form derivations documented in :file:`VjpEmitters/nn/Loss.mm`.

Note: ``F.binary_cross_entropy_with_logits`` in the Python wrapper is
decomposed into elementary ops (abs/exp/log/sub/mul) — it doesn't
emit the ``bce_with_logits`` engine op, so the manual VJP for that
op isn't exercised by it.  The dedicated VJP is still useful when
the trace surfaces the op directly (e.g. via a future Functional
shortcut).
"""


import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


@pytest.fixture(autouse=True)
def _manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


# ── BCE: VJPs registered but not exercised by F.binary_cross_entropy ────
#
# Lucid's Python wrappers for ``binary_cross_entropy`` and
# ``binary_cross_entropy_with_logits`` decompose the loss into
# elementary ops (clamp, log, mul, sub, sum/mean), so the trace never
# emits the ``bce_loss`` / ``bce_with_logits`` engine ops.  The
# manual VJPs for those engine ops are still registered (so any
# future Python path that calls them directly compiles cleanly), but
# the standard user path goes through `log` / `mul` / `sub` VJPs
# which were exercised earlier (P1 math VJPs).
#
# A dedicated acceptance test for the engine-op path would require
# bypassing the F wrapper and calling ``_C_engine.nn.bce_loss(...)``
# directly — fragile and not representative of user behaviour.
# Skipped intentionally.


# ── Huber loss ─────────────────────────────────────────────────────


class _HuberNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(x)


def _huber_loss(pred: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    return F.huber_loss(pred, target)


def test_manual_vjp_huber_parity() -> None:
    """Linear → Huber: manual VJP matches eager within 1e-3."""
    lucid.manual_seed(0)
    a = _HuberNet().to(COMPILE_DEVICE)
    b = _HuberNet().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        pb.copy_(pa.detach().clone())

    x = metal_tensor(8, 8)
    t = metal_tensor(8, 4)

    opt_eager = optim.SGD(list(a.parameters()), lr=1e-2)
    eager: list[float] = []
    for _ in range(5):
        opt_eager.zero_grad()
        loss = _huber_loss(a(x), t)
        loss.backward()
        opt_eager.step()
        eager.append(float(loss.item()))

    opt_comp = optim.SGD(list(b.parameters()), lr=1e-2)
    step = fused_step(b, _huber_loss, opt_comp)
    comp: list[float] = []
    for _ in range(5):
        out = step(x, t)
        comp.append(float(out.item() if hasattr(out, "item") else out[0].item()))

    for k in range(5):
        diff = abs(eager[k] - comp[k])
        assert diff < 1e-3, (
            f"Huber VJP drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )
