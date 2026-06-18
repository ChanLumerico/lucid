"""Dynamic batch: per-shape static (default) + experimental symbolic-batch (opt-in).

``lucid.compile(model, dynamic=True)`` declares that the module will be called
with varying batch sizes.  Two regimes:

* **Default** — robust *per-shape static caching*: one executable per distinct
  input shape, cached and reused.  Works for every model, never crashes; calling
  at K distinct batch sizes leaves K cache entries.

* **``LUCID_COMPILE_DYNAMIC=1`` (experimental)** — a single *symbolic-batch*
  executable shared across batch sizes (one cache entry for all of them).  This
  rides on the dynamic-batch-aware view emitters (reshape / flatten / squeeze /
  reduce-squeeze keep the symbolic batch at dim 0), which is what lets real
  transformers (attention head-split + SDPA + merge) and CNNs (conv + flatten)
  share one executable instead of aborting MPSGraph's MLIR pass.

The compiled training step (``make_step``) is always per-shape static — the
backward graph of common reductions aborts under a symbolic batch axis.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import make_step
from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_cache_hit,
    metal_tensor,
    to_metal,
)


def _maxdiff(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


class TestDynamicDefaultIsStatic:
    """``dynamic=True`` WITHOUT the env opt-in == safe per-shape static caching."""

    def test_transformer_per_shape_no_crash(self) -> None:
        m = to_metal(
            nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, batch_first=True)
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        assert cm._symbolic is False  # no env opt-in → static
        for bs in (2, 4):
            x = metal_tensor(bs, 5, 32)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert_cache_hit(cm, 2)  # one executable per distinct shape

    def test_conv_per_shape_no_crash(self) -> None:
        m = to_metal(
            nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Flatten())
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4):
            x = metal_tensor(bs, 3, 8, 8)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert_cache_hit(cm, 2)


class TestSymbolicBatchOptIn:
    """``LUCID_COMPILE_DYNAMIC=1``: ONE executable shared across batch sizes."""

    @pytest.fixture(autouse=True)
    def _opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LUCID_COMPILE_DYNAMIC", "1")

    def test_mlp_one_shared_executable(self) -> None:
        m = to_metal(
            nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        assert cm._symbolic is True
        for bs in (8, 16, 32):
            x = metal_tensor(bs, 16)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert_cache_hit(cm, 1)  # SHARED across every batch size

    def test_transformer_one_shared_executable(self) -> None:
        # The keystone: the dynamic-batch-aware reshape emitter lets a real
        # transformer (attention head-split + SDPA + head-merge + LayerNorm +
        # FFN + residual) share a single symbolic-batch executable.
        m = to_metal(
            nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, batch_first=True)
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4, 8):
            x = metal_tensor(bs, 7, 32)
            assert _maxdiff(cm(x), m(x)) < 1e-3
        assert_cache_hit(cm, 1)

    def test_cnn_one_shared_executable(self) -> None:
        # conv (symbolic-safe) + flatten (dynamic-batch-aware reshape) + linear.
        m = to_metal(
            nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(8, 10),
            )
        ).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4):
            x = metal_tensor(bs, 3, 8, 8)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert_cache_hit(cm, 1)

    def test_scalar_arithmetic_symbolic(self) -> None:
        # Scalar broadcast (`x*0.5`, `1.0-x`, `(x-μ)/2`) rides symbolic batch:
        # the 0-dim scalar promotion (+ comparison/bitwise broadcast) keeps the
        # batch symbolic instead of pinning it via a full-shape constant.
        class _Net(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                y = (x - mu) * 0.5 + 1.0
                return lucid.where(y > 0.0, y, y * 0.25)

        m = to_metal(_Net()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4, 8):
            x = metal_tensor(bs, 6, 16)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert_cache_hit(cm, 1)

    def test_off_dim0_view_falls_back_not_corrupts(self) -> None:
        # A view that moves the batch OFF dim 0 (unsqueeze-at-front) cannot be
        # represented under a dim-0 symbolic batch.  The emitter must bail so the
        # compile falls back (correct results) rather than silently mis-shaping
        # the graph — verify correctness holds across batch sizes (the failure
        # mode this guards against is silent wrong numerics, not a crash).
        class _OffDim0(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                y = x.unsqueeze(0)  # (B, F) -> (1, B, F): batch slides to dim 1
                y = y + 1.0
                return y.squeeze(0)  # back to (B, F)

        m = to_metal(_OffDim0()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4, 8):
            x = metal_tensor(bs, 5)
            assert _maxdiff(cm(x), m(x)) < 1e-5  # correct at every batch size

    def test_reduce_squeeze_symbolic(self) -> None:
        # mean over non-batch axes (keepdim=False squeeze) keeps the symbolic
        # batch — exercises the reduce-squeeze dynamic-aware path directly.
        class _Pool(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c = nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.c(x).mean(dim=(2, 3))

        m = to_metal(_Pool()).eval()
        cm = lucid.compile(m, dynamic=True)
        for bs in (2, 4):
            x = metal_tensor(bs, 3, 8, 8)
            assert _maxdiff(cm(x), m(x)) < 1e-4
        assert_cache_hit(cm, 1)


class TestMakeStepDynamicIsStatic:
    """``make_step(dynamic=True)`` is always per-shape static (backward unsafe)."""

    def test_cross_entropy_no_crash(self) -> None:
        m = to_metal(nn.Linear(16, 4))
        step = make_step(m, lambda y, t: F.cross_entropy(y, t), dynamic=True)
        for bs in (8, 16):
            x = metal_tensor(bs, 16)
            t = lucid.zeros(bs, dtype=lucid.int64).to(COMPILE_DEVICE)
            loss = step(x, t)
            loss.backward()
        assert len(step.cache) == 2  # one fwd+bwd executable per distinct shape
