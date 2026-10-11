"""bfloat16 lowers to MPSGraph, and the compiled answer is eager's (LCD-327).

The compile dtype table (``MpsDtype.h``) had no bfloat16 entry, so every
bfloat16 call was refused before the trace and ran eagerly.  MPSGraph holds
``MPSDataTypeBFloat16``; each case here runs compiled — no fallback
warning, a cached executable — and agrees with eager to within bfloat16's
rounding: both compute at reduced precision, and their order of operations
differs, so a result may land one bfloat16 step apart (2^-7 relative).
"""

import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
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
