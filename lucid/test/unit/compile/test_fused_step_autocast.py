"""fused_step + autocast — X4.4 acceptance.

The autocast → fused_step integration works as a side-effect of:

1. C++ ``AutocastGuard`` already dispatches per-op forward
   (so the trace records ``astype`` casts at op boundaries when
   the trace runs inside ``with autocast()``)
2. The ``astype`` VJP from Manual VJP P1 differentiates those
   casts cleanly (gradient flows back as ``astype`` to the
   source dtype)
3. The mixed-dtype trace acceptance lifted in X4.1
4. The compile path's split-scope fix in
   ``_FusedStep._build_executable`` — capture user's autocast
   state, run model + loss under it, but disable for the
   optimizer ``_trace_update`` (so F32 master weights stay F32
   and don't get cast to F16 by the optimizer reading them
   through the autocast guard)

The whole X4.4 work landed in ~25 lines of `_fused_step.py`
(vs ~300 estimated in [[retro-3-5-phase-vjp-long-tail]]) because
items 1-3 above were already in place from prior phases.

Known limitations (documented for follow-up):
- BatchNorm under autocast in fused_step fails (`emitter
  'batch_norm' returned false`).  The BN op's AmpPolicy is
  ``KeepInput`` (vs ForceFP32 in PyTorch) which casts BN inputs
  to F16 — the compile-path BN emitter doesn't support F16
  inputs.  Worked around in production by using LayerNorm
  (transformer pattern) or by manually disabling autocast around
  BN modules.  Tracked as a separate AMP coverage gap.
- GradScaler (X4.3) — not integrated; the basic mixed-dtype
  fused_step is numerically stable for many models without it,
  but production AMP training typically needs it for F16
  gradient underflow protection.  Tracked as P2.2.
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import lucid.amp as amp
import lucid.metal as metal
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _loss_fn(pred: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    """MSE in F32 — cast pred up from F16 to match target dtype."""
    return F.mse_loss(pred.to(target.dtype), target)


# ── Simple MLP — base case ──────────────────────────────────────────


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.fc1(x).relu())


def test_fused_step_autocast_mlp_trains() -> None:
    """3-layer MLP + fused_step + autocast(F16) — 5 SGD steps converge."""
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    losses: list[float] = []
    for _ in range(5):
        with amp.autocast(dtype=lucid.float16):
            loss = step(x, t)
        _ = float(loss.item())
        metal.synchronize()
        losses.append(float(loss.item()))

    assert losses[0] > losses[-1], (
        f"expected loss to decrease over 5 SGD steps; got {losses}"
    )


def test_fused_step_autocast_preserves_f32_master_weights() -> None:
    """Critical invariant: model parameters stay F32 after autocast steps.

    The split-scope fix in `_FusedStep._build_executable` is what
    enforces this — without it the optimizer would read the params
    through the active autocast guard, cast them to F16, and the
    new_param output would be F16 (causing a dtype-mismatch error
    in `run_executable_inplace`'s buffer-binding step).
    """
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    with amp.autocast(dtype=lucid.float16):
        step(x, t)
    metal.synchronize()

    for name, p in model.named_parameters():
        assert p.dtype == lucid.float32, (
            f"param {name} got cast to {p.dtype} under autocast; "
            "master weights must stay F32 — the split-scope fix in "
            "_FusedStep._build_executable failed."
        )


# ── LayerNorm-MLP — transformer-shape ───────────────────────────────


class _LNMLP(nn.Module):
    """LayerNorm-based MLP — exercises the LN path under autocast.

    LayerNorm's compile-path emitter handles F16 inputs cleanly
    (unlike the BatchNorm emitter which currently rejects F16 — see
    module docstring §"Known limitations").  This shape is the
    standard "pre-norm transformer block without attention" pattern.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 32)
        self.ln2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc3(self.ln2(self.fc2(self.ln1(self.fc1(x))).relu()))


def test_fused_step_autocast_layernorm_mlp_trains() -> None:
    """LN-based MLP + autocast(F16) + fused_step — 5 SGD steps converge."""
    lucid.manual_seed(0)
    model = _LNMLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    losses: list[float] = []
    for _ in range(5):
        with amp.autocast(dtype=lucid.float16):
            loss = step(x, t)
        _ = float(loss.item())
        metal.synchronize()
        losses.append(float(loss.item()))

    assert losses[0] > losses[-1], (
        f"expected loss to decrease over 5 SGD steps; got {losses}"
    )


# ── Forward-only ── `lucid.compile(model)` + autocast ──────────────


def test_lucid_compile_forward_autocast_matches_eager() -> None:
    """lucid.compile(model) + autocast(F16) — output parity vs eager."""
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.eval()
    x = lucid.randn(2, 8).to(COMPILE_DEVICE)

    with amp.autocast(dtype=lucid.float16):
        out_eager = model(x)
    _ = float(out_eager.sum().item())
    metal.synchronize()

    cm = lucid.compile(model)
    with amp.autocast(dtype=lucid.float16):
        out_compile = cm(x)
    _ = float(out_compile.sum().item())
    metal.synchronize()

    assert out_eager.dtype == lucid.float16
    assert out_compile.dtype == lucid.float16

    abs_diff = float((out_eager - out_compile).abs().max().item())
    scale = float(out_eager.abs().max().item())
    rel_diff = abs_diff / max(scale, 1e-9)
    # F16 tolerance — 1e-2 is the standard "within ~3 decimal digits" gate.
    assert rel_diff < 1e-2, (
        f"compile output diverges from eager under autocast; "
        f"rel_diff={rel_diff:.4e}"
    )
