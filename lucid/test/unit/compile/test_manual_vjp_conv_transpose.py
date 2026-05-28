"""Manual VJP — Conv-transpose + Conv3d acceptance (X1.3).

Two small tests:
  * ConvTranspose2d as a tiny generator block.
  * Conv3d on a (B, C, D, H, W) volume.

Both verify forward + backward + bias gradient match eager under
``LUCID_MANUAL_VJP_REQUIRE=1`` (manual-or-die — proves the new
VJPs are actually executing).
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


# ── ConvTranspose2d ─────────────────────────────────────────────────


class _ConvTransposeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # in=4, out=2, kernel=2, stride=2 → doubles spatial size.
        self.up = nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2)
        self.fc = nn.Linear(2, 1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.up(x).relu()
        h = h.mean(dim=(2, 3), keepdim=False)  # (B, 2)
        return self.fc(h)


def _loss(p: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(p, t)


def test_manual_vjp_conv_transpose2d_parity() -> None:
    """ConvTranspose2d + ReLU + mean + Linear + MSE: ≤ 2e-3 over 3 SGD steps."""
    lucid.manual_seed(0)
    a = _ConvTransposeNet().to(COMPILE_DEVICE)
    b = _ConvTransposeNet().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        pb.copy_(pa.detach().clone())

    x = metal_tensor(2, 4, 3, 3)
    t = metal_tensor(2, 1)

    opt_eager = optim.SGD(list(a.parameters()), lr=1e-2)
    eager: list[float] = []
    for _ in range(3):
        opt_eager.zero_grad()
        loss = _loss(a(x), t)
        loss.backward()
        opt_eager.step()
        eager.append(float(loss.item()))

    opt_comp = optim.SGD(list(b.parameters()), lr=1e-2)
    step = fused_step(b, _loss, opt_comp)
    comp: list[float] = []
    for _ in range(3):
        out = step(x, t)
        comp.append(float(out.item() if hasattr(out, "item") else out[0].item()))

    for k in range(3):
        diff = abs(eager[k] - comp[k])
        assert diff < 2e-3, (
            f"ConvTranspose2d VJP drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )


# ── Conv3d ──────────────────────────────────────────────────────────


class _Conv3dNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(2, 3, kernel_size=2, padding=0)
        self.fc = nn.Linear(3, 1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.conv(x).relu()
        h = h.mean(dim=(2, 3, 4), keepdim=False)  # (B, 3)
        return self.fc(h)


def test_manual_vjp_conv3d_parity() -> None:
    """Conv3d + ReLU + mean + Linear + MSE: ≤ 2e-3 over 3 SGD steps."""
    lucid.manual_seed(0)
    a = _Conv3dNet().to(COMPILE_DEVICE)
    b = _Conv3dNet().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        pb.copy_(pa.detach().clone())

    x = metal_tensor(2, 2, 3, 3, 3)
    t = metal_tensor(2, 1)

    opt_eager = optim.SGD(list(a.parameters()), lr=1e-2)
    eager: list[float] = []
    for _ in range(3):
        opt_eager.zero_grad()
        loss = _loss(a(x), t)
        loss.backward()
        opt_eager.step()
        eager.append(float(loss.item()))

    opt_comp = optim.SGD(list(b.parameters()), lr=1e-2)
    step = fused_step(b, _loss, opt_comp)
    comp: list[float] = []
    for _ in range(3):
        out = step(x, t)
        comp.append(float(out.item() if hasattr(out, "item") else out[0].item()))

    for k in range(3):
        diff = abs(eager[k] - comp[k])
        assert diff < 2e-3, (
            f"Conv3d VJP drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )
