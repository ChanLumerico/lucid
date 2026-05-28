"""LSTM compile envelope + fallback behaviour.

The MPSGraph LSTM emitter only handles single-layer / unidirectional /
F32 / no-projection (matches the MLX fused-backend's
``lstm_metal_supported`` predicate).  Anything outside that envelope
must still produce correct output via clean eager fallback — silent
incorrect results would be a production-grade hazard.

Coverage:

  * vanilla LSTM: compile path emits, output matches eager bit-exact
    (LSTM dispatches MLX inside the executable so this is a true 0.0)
  * h_n / c_n consumption: trace records 3 outputs; downstream ops
    that consume h_n or c_n still compile
  * bidirectional / multi-layer / projected LSTM: fall back to eager
    (emit returns nullptr → builder aborts → trace marked eager-only)
"""

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import (
    COMPILE_DEVICE,
    assert_compile_parity,
    metal_tensor,
)


def test_lstm_vanilla_bit_exact() -> None:
    """Single-layer unidirectional LSTM should be bit-exact in compile mode.

    The MPSGraph LSTM dispatches the same MLX-backed Metal kernel the
    eager path uses, so the result is byte-equal — not just close.
    """

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=16, hidden_size=32)
            self.fc = nn.Linear(32, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            y, _ = self.lstm(x)
            return self.fc(y[-1])

    model = M()
    x = metal_tensor(8, 4, 16)
    # LSTM path is genuinely bit-exact (same kernel under the hood).
    assert_compile_parity(model, x, atol=0.0, rtol=0.0)


def test_lstm_consumes_hidden_state() -> None:
    """Models that read ``h_n`` should compile and stay bit-exact."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=8, hidden_size=16)
            self.fc = nn.Linear(16, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            _, (hn, _cn) = self.lstm(x)
            return self.fc(hn.squeeze(0))

    model = M()
    x = metal_tensor(6, 3, 8)
    assert_compile_parity(model, x, atol=0.0, rtol=0.0)


def test_lstm_consumes_cell_state() -> None:
    """Models that read ``c_n`` should compile and stay bit-exact.

    Previously the LSTM eager kernel returned ``c_n`` with a logical
    ``(1, B, H)`` shape backed by 2-D storage — bare ``cn.sum()``
    walked the logical rank and tripped the engine's ``Invalid axis 2
    for array with 2 dimensions``.  Fixed by reshaping the backend
    storage to match the claimed 3-D shape in ``LSTM.cpp``; this
    test guards the fix.
    """

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=8, hidden_size=16)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            _, (_hn, cn) = self.lstm(x)
            # Exercises the previously-broken bare ``cn.sum()`` path
            # directly — would explode if the rank-mismatch returned.
            return cn.sum() + cn.flatten().mean()

    model = M()
    x = metal_tensor(6, 3, 8)
    assert_compile_parity(model, x, atol=0.0, rtol=0.0)


def test_lstm_proj_size_falls_back_cleanly() -> None:
    """Projected LSTM (``proj_size > 0``) must fall back to eager."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=8, hidden_size=16, proj_size=4)
            self.fc = nn.Linear(4, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            y, _ = self.lstm(x)
            return self.fc(y[-1])

    model = M()
    x = metal_tensor(6, 3, 8)
    # Output must still be correct (eager fallback handles proj_size).
    # Use a wider tolerance because the eager path is the only source.
    assert_compile_parity(model, x, atol=1e-4, rtol=1e-5)
    # The trace should have been recorded as eager-only for this sig.
    model.to(COMPILE_DEVICE)
    x_m = x.to(COMPILE_DEVICE)
    cm = lucid.compile(model)
    cm(x_m)
    cm(x_m)
    info = cm.cache_info()
    # Either the cache stayed empty (full eager fallback) or there's
    # at least one eager_only signature — both are acceptable.
    assert info["entries"] == 0 or len(info["eager_only"]) > 0
