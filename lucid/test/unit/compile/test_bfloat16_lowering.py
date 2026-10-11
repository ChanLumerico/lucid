"""bfloat16 lowers to MPSGraph, and the compiled answer is eager's (LCD-327).

The compile dtype table (``MpsDtype.h``) had no bfloat16 entry, so every
bfloat16 call was refused before the trace and ran eagerly.  MPSGraph holds
``MPSDataTypeBFloat16``; each case here runs compiled — no fallback
warning, a cached executable — and agrees with eager to within bfloat16's
rounding: both compute at reduced precision, and their order of operations
differs, so a result may land one bfloat16 step apart (2^-7 relative).
The training cases hold ``make_step``'s gradients and three ``fused_step``
SGD updates to two bfloat16 steps of eager's.
"""

import math
import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

#: Two bfloat16 steps at the magnitude of the result.
REL = 2.0**-6


class _Apply(nn.Module):
    def __init__(self, fn: Callable[[nn.Module, lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.ln = nn.LayerNorm(8)
        self.emb = nn.Embedding(10, 8)
        self._fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self._fn(self, x)


def _ids(x: lucid.Tensor) -> lucid.Tensor:
    """Row indices for the embedding table, built from ``x``'s sign."""
    return (x[:, :3] > 0).to(lucid.int32) + 2


CASES: dict[str, Callable[[nn.Module, lucid.Tensor], lucid.Tensor]] = {
    "linear": lambda m, x: m.lin(x),
    "matmul": lambda m, x: x @ x.transpose(0, 1),
    "softmax": lambda m, x: F.softmax(x, dim=-1),
    "layer_norm": lambda m, x: m.ln(x),
    "embedding": lambda m, x: m.emb(_ids(x)),
    "one_hot": lambda m, x: F.one_hot(_ids(x).to(lucid.int64), 10).to(lucid.bfloat16),
    "gelu": lambda m, x: F.gelu(x),
    "sum": lambda m, x: x.sum(dim=-1),
    "mlp": lambda m, x: m.ln(F.gelu(m.lin(x)) + x),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_bfloat16_call_compiles_and_matches_eager(case: str) -> None:
    lucid.manual_seed(0)
    model = _Apply(CASES[case]).to(COMPILE_DEVICE).to(lucid.bfloat16).eval()
    x = lucid.randn(4, 8).to(COMPILE_DEVICE).to(lucid.bfloat16)
    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        compiled(x)
        got = compiled(x)
    assert not compiled.cache_info()["eager_only"]
    want = model(x)
    assert got.dtype == lucid.bfloat16 == want.dtype
    assert got.shape == want.shape
    g, w = got.to(lucid.float32), want.to(lucid.float32)
    bound = REL * max(1.0, float(w.abs().max().item()))
    assert float((g - w).abs().max().item()) <= bound


# ── training: gradients and an optimizer step at bfloat16 ───────────


def _two_steps(t: lucid.Tensor) -> float:
    """Two bfloat16 steps at ``t``'s largest magnitude, rounded up to a power
    of two: the gap a different order of operations may leave."""
    peak = max(float(t.abs().max().item()), 2.0**-100)
    return 2.0 * 2.0 ** (math.ceil(math.log2(peak)) - 8)


def _mlp() -> nn.Module:
    lucid.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16), nn.GELU(), nn.LayerNorm(16), nn.Linear(16, 4)
    )
    return model.to(COMPILE_DEVICE).to(lucid.bfloat16)


def _batch() -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    x = lucid.randn(6, 8).to(COMPILE_DEVICE).to(lucid.bfloat16)
    t = lucid.randn(6, 4).to(COMPILE_DEVICE).to(lucid.bfloat16)
    return x, t


def _close(got: lucid.Tensor, want: lucid.Tensor) -> None:
    assert got.dtype == lucid.bfloat16 == want.dtype
    g, w = got.to(lucid.float32), want.to(lucid.float32)
    assert float((g - w).abs().max().item()) <= _two_steps(w)


def test_make_step_gradients_match_eager_at_bfloat16() -> None:
    model = _mlp()
    x, t = _batch()
    F.mse_loss(model(x), t).backward()
    eager = [p.grad.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = None
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        lucid.compile.make_step(model, F.mse_loss)(x, t).backward()
    for want, p in zip(eager, model.parameters(), strict=True):
        assert p.grad is not None
        _close(p.grad, want)


def test_fused_step_sgd_matches_eager_at_bfloat16() -> None:
    eager, compiled = _mlp(), _mlp()
    x, t = _batch()
    eager_opt = optim.SGD(eager.parameters(), lr=0.1)
    step = fused_step(compiled, F.mse_loss, optim.SGD(compiled.parameters(), lr=0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        for _ in range(3):
            eager_opt.zero_grad()
            F.mse_loss(eager(x), t).backward()
            eager_opt.step()
            step(x, t)
    for want, got in zip(eager.parameters(), compiled.parameters(), strict=True):
        _close(got, want)
