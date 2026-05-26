"""Manual VJP — ``gelu_exact`` (erf-based GELU) regression.

The compile-side ``GeluExactVjp`` differentiates the exact GELU
formula

    y = 0.5 * x * (1 + erf(x / sqrt(2)))

with the chain rule

    dy/dx = 0.5 * (1 + erf(x/√2))  +  x * exp(-x²/2) / sqrt(2π).

This file covers two surfaces:

1.  ``test_manual_vjp_gelu_exact_transformer_parity`` — fp32 loss
    parity vs eager on a small transformer-block-shaped network
    that uses ``F.gelu(approximate="none")``.  Catches a wrong
    derivative formula (sign / constant) — those produce O(1)
    drift after a single SGD step.

2.  ``test_manual_vjp_gelu_exact_autocast_trains`` — the same
    network under ``autocast(F16)``.  Exercises the mixed-dtype
    reconciliation that ``emit_unary_vjp`` provides: ``x`` is
    pre-cast to ``go.dataType`` so the in-body constants (all
    resolved against ``go.dataType``) align with both operands of
    every multiply / add.  If a constant accidentally derived from
    ``x.dataType`` (the F32 master) the F16 chain would trip
    MPSGraph's same-dtype check and ``fused_step`` would raise.
"""

import os

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import lucid.amp as amp
import lucid.metal as metal
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

BATCH = 2
SEQ = 4
DIM = 8
OUT = 4


class _GeluBlock(nn.Module):
    """LayerNorm → Linear → GELU(exact) → Linear → mean-pool → Linear.

    The ``F.gelu(approximate="none")`` call is the only path that
    exercises ``gelu_exact`` (the ``"tanh"`` variant routes to the
    separate ``gelu`` op + ``GeluVjp``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(DIM)
        self.fc1 = nn.Linear(DIM, 4 * DIM)
        self.fc2 = nn.Linear(4 * DIM, DIM)
        self.proj = nn.Linear(DIM, OUT)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.ln(x)                                  # (B, T, D)
        h = self.fc2(F.gelu(self.fc1(h), approximate="none"))  # (B, T, D)
        pooled = h.mean(dim=1, keepdim=False)           # (B, D)
        return self.proj(pooled)                        # (B, OUT)


def _loss_fn(pred: lucid.Tensor, tgt: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(pred, tgt)


def _autocast_loss_fn(pred: lucid.Tensor, tgt: lucid.Tensor) -> lucid.Tensor:
    # Cast F16 pred back up to match the F32 target before MSE.
    return F.mse_loss(pred.to(tgt.dtype), tgt)


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _GeluBlock().to(COMPILE_DEVICE)
    b = _GeluBlock().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        pb.copy_(pa.detach().clone())
    return a, b


def _eager_trajectory(model, x, t, steps, lr):
    opt = optim.SGD(list(model.parameters()), lr=lr)
    out = []
    for _ in range(steps):
        opt.zero_grad()
        loss = _loss_fn(model(x), t)
        loss.backward()
        opt.step()
        out.append(float(loss.item()))
    return out


def _compile_trajectory(model, x, t, steps, lr):
    opt = optim.SGD(list(model.parameters()), lr=lr)
    step = fused_step(model, _loss_fn, opt)
    out = []
    for _ in range(steps):
        loss = step(x, t)
        out.append(float(loss.item() if hasattr(loss, "item") else loss[0].item()))
    return out


@pytest.fixture(autouse=True)
def _manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


def test_manual_vjp_gelu_exact_transformer_parity() -> None:
    """5 SGD steps on the GELU-block: manual-VJP loss matches eager ≤ 2e-3."""
    lucid.manual_seed(0)
    x = metal_tensor(BATCH, SEQ, DIM)
    t = metal_tensor(BATCH, OUT)
    eager_model, comp_model = _matched_models()

    eager = _eager_trajectory(eager_model, x, t, steps=5, lr=1e-2)
    comp = _compile_trajectory(comp_model, x, t, steps=5, lr=1e-2)

    assert len(eager) == len(comp) == 5
    for k in range(5):
        diff = abs(eager[k] - comp[k])
        assert diff < 2e-3, (
            f"gelu_exact manual VJP drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )


def test_manual_vjp_gelu_exact_autocast_trains() -> None:
    """GELU-block + ``fused_step`` + ``autocast(F16)`` — 5 SGD steps converge.

    The F16 chain forces the VJP's constants + the (pre-cast) forward
    ``x`` to share a dtype.  A regression where ``GeluExactVjp``
    derives any constant from ``x.dataType`` (the F32 master) instead
    of ``go.dataType`` would surface here as an MPSGraph dtype
    assertion at ``compile_generic_fused_step`` time.
    """
    lucid.manual_seed(0)
    model = _GeluBlock().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-2)
    step = fused_step(model, _autocast_loss_fn, opt)

    x = lucid.randn(BATCH, SEQ, DIM).to(COMPILE_DEVICE)
    t = lucid.randn(BATCH, OUT).to(COMPILE_DEVICE)

    losses: list[float] = []
    for _ in range(5):
        with amp.autocast(dtype=lucid.float16):
            loss = step(x, t)
        metal.synchronize()
        losses.append(float(loss.item()))

    assert losses[0] > losses[-1], (
        f"expected gelu_exact block loss to decrease over 5 SGD steps under "
        f"autocast(F16) + LUCID_MANUAL_VJP_REQUIRE=1; got {losses}"
    )


def test_manual_vjp_gelu_exact_env_seen() -> None:
    assert os.environ.get("LUCID_MANUAL_VJP") == "1"
    assert os.environ.get("LUCID_MANUAL_VJP_REQUIRE") == "1"
