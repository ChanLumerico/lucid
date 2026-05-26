"""Manual VJP — transformer-block-shaped acceptance test.

Exercises Split + Cat + LayerNorm + Linear + Softmax VJPs together
on a mini-transformer-block-shaped network.  This is the surface that
MPSGraph autograd asserts on (sliceTensor: from split, plus the
gatherWithUpdatesTensor: if embedding is added — and even without
embedding, the split + LN chain alone triggers
``Not a predecessor of primaryTensor``).

Network (intentionally small):
  x (B, T, D)
  ↓ LayerNorm
  ↓ Linear(D → 3*D)          ← projects to Q|K|V concatenated
  ↓ split into 3 pieces of size D on last dim
  ↓ cat back along last dim   ← exercise of the inverse pair
  ↓ Linear(D → out_dim)
  ↓ softmax(dim=-1)
  ↓ MSE against random target

Backward path touches: linear (×2), softmax, split, cat, layer_norm,
mse_loss, mean (via the linear flatten), reshape (linear internal).

Pass criterion under ``LUCID_MANUAL_VJP_REQUIRE=1``: 5-step SGD loss
within ``2e-3`` of eager (slightly looser than Phase 1-4 tolerance —
LayerNorm + split chain accumulates fp32 reordering noise faster).
"""

import os

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

BATCH = 2
SEQ = 4
DIM = 8
OUT = 4


class _MiniBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(DIM)
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.proj = nn.Linear(DIM, OUT)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.ln(x)  # (B, T, D)
        qkv = self.qkv(h)  # (B, T, 3D)
        parts = lucid.split(qkv, DIM, dim=-1)  # list of 3 (B, T, D)
        merged = lucid.cat(parts, dim=-1)  # (B, T, 3D)
        # Reduce to per-batch logits via average over seq + linear proj
        # on just the first slice (any of the three would do — same shape).
        sliced = lucid.split(merged, DIM, dim=-1)[0]  # (B, T, D)
        pooled = sliced.mean(dim=1, keepdim=False)  # (B, D)
        logits = self.proj(pooled)  # (B, OUT)
        return F.softmax(logits, dim=-1)


def _loss_fn(pred: lucid.Tensor, tgt: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(pred, tgt)


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _MiniBlock().to(COMPILE_DEVICE)
    b = _MiniBlock().to(COMPILE_DEVICE)
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


def test_manual_vjp_transformer_block_parity() -> None:
    """LayerNorm + Linear + split + cat + Linear + softmax + MSE under manual VJP."""
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
            f"transformer-block manual VJP drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )


def test_manual_vjp_env_seen() -> None:
    assert os.environ.get("LUCID_MANUAL_VJP") == "1"
    assert os.environ.get("LUCID_MANUAL_VJP_REQUIRE") == "1"
