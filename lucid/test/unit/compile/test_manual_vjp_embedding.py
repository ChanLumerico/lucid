"""Manual VJP — Embedding/Softmax acceptance test.

This is the canonical "MPSGraph autograd asserts but manual VJP works"
demonstration.  The forward path uses ``embedding`` (which lowers to
MPSGraph's ``gatherWithUpdatesTensor:indicesTensor:``) — the *exact*
op that triggers ``"Not a predecessor of primaryTensor"`` when
fed into ``gradientForPrimaryTensor:``.

With ``LUCID_MANUAL_VJP=1 LUCID_MANUAL_VJP_REQUIRE=1`` the test must
pass (loss parity with eager); without manual VJP, ``fused_step``
would either abort or produce wrong gradients on the embedding
weight matrix.

Network:
  idx (B, N) → embed (V, D) → mean over seq dim → Linear → softmax → NLL-style sum loss

Backward path exercises: embedding VJP (scatter-add) → mean VJP →
linear VJP → softmax VJP → sum VJP.
"""

import os

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

VOCAB = 24
EMB_DIM = 8
OUT_DIM = 4
BATCH = 4
SEQ = 6


class _EmbedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMB_DIM)
        self.fc = nn.Linear(EMB_DIM, OUT_DIM)

    def forward(self, idx: lucid.Tensor) -> lucid.Tensor:
        h = self.emb(idx)  # (B, N, D)
        h = h.mean(dim=1, keepdim=False)  # (B, D)
        h = self.fc(h)  # (B, OUT_DIM)
        return F.softmax(h, dim=-1)  # (B, OUT_DIM)


def _embed_loss(pred: lucid.Tensor, tgt: lucid.Tensor) -> lucid.Tensor:
    # MSE on the softmax output against a one-hot target — same diff
    # structure as cross-entropy but avoids touching the log_softmax
    # path for this slice of the surface.
    return F.mse_loss(pred, tgt)


def _matched_models() -> tuple[nn.Module, nn.Module]:
    lucid.manual_seed(0)
    a = _EmbedNet().to(COMPILE_DEVICE)
    b = _EmbedNet().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        pb.copy_(pa.detach().clone())
    return a, b


def _eager_trajectory(
    model: nn.Module, idx: lucid.Tensor, t: lucid.Tensor, steps: int, lr: float
) -> list[float]:
    opt = optim.SGD(list(model.parameters()), lr=lr)
    out: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = _embed_loss(model(idx), t)
        loss.backward()
        opt.step()
        out.append(float(loss.item()))
    return out


def _compile_trajectory(
    model: nn.Module, idx: lucid.Tensor, t: lucid.Tensor, steps: int, lr: float
) -> list[float]:
    opt = optim.SGD(list(model.parameters()), lr=lr)
    step = fused_step(model, _embed_loss, opt)
    out: list[float] = []
    for _ in range(steps):
        loss = step(idx, t)
        out.append(float(loss.item() if hasattr(loss, "item") else loss[0].item()))
    return out


@pytest.fixture(autouse=True)
def _manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


def test_manual_vjp_embedding_loss_parity() -> None:
    """Embedding → mean → Linear → softmax → MSE: manual VJP matches eager."""
    lucid.manual_seed(0)
    # int32 indices in [0, VOCAB).
    idx = lucid.randint(0, VOCAB, size=(BATCH, SEQ)).to(COMPILE_DEVICE)
    t = metal_tensor(BATCH, OUT_DIM)
    eager_model, comp_model = _matched_models()

    eager = _eager_trajectory(eager_model, idx, t, steps=5, lr=1e-2)
    comp = _compile_trajectory(comp_model, idx, t, steps=5, lr=1e-2)

    assert len(eager) == len(comp) == 5
    for k in range(5):
        diff = abs(eager[k] - comp[k])
        assert diff < 1e-3, (
            f"embedding manual VJP drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )


def test_manual_vjp_env_seen() -> None:
    """Confirm both env vars are visible inside the test."""
    assert os.environ.get("LUCID_MANUAL_VJP") == "1"
    assert os.environ.get("LUCID_MANUAL_VJP_REQUIRE") == "1"
