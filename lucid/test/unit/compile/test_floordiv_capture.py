"""Integer ``//`` has to reach the graph, and still floor once it does.

Two defects met here, and each hid the other.

``floordiv`` on integers bypasses ``wire_autograd`` — it has no
derivative — and that is also what records a traced op's operands. The
op therefore arrived in the trace with no inputs, and the tracer refused
the whole graph rather than bake a value it could not prove constant.
Every model whose positional encoding derives an index by division fell
back to eager without saying so: three-axis RoPE computes
``ids // (height * width)`` for its depth axis, so the entire V-JEPA and
V-JEPA 2 family ran uncompiled.

Because the op never reached the emitter, the emitter had never run. It
was ``floor(a / b)``, which is right for floats and wrong for integers:
MPSGraph's integer division truncates toward zero and flooring an
integer changes nothing, so ``-7 // 2`` would have come back ``-3``.
Fixing only the tracing would have replaced a silent fallback with a
silent wrong answer, which is worse — the negative cases below are the
ones that catch it.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

#: Both signs on both operands, an exact division, and a quotient that
#: rounds the wrong way under truncation (``-1 // 5`` is ``-1``, not 0).
CASES = [(7, 2), (-7, 2), (7, -2), (-7, -2), (8, 4), (-8, 4), (0, 3), (-1, 5)]


class _Floordiv(nn.Module):
    def forward(self, a: lucid.Tensor, b: lucid.Tensor) -> lucid.Tensor:
        return (a // b).to(lucid.float32)


def _operands() -> tuple[lucid.Tensor, lucid.Tensor]:
    a = lucid.tensor([p[0] for p in CASES]).to(lucid.int64).to(COMPILE_DEVICE)
    b = lucid.tensor([p[1] for p in CASES]).to(lucid.int64).to(COMPILE_DEVICE)
    return a, b


def test_integer_floordiv_is_captured() -> None:
    """The signature must land in the cache, not in ``eager_only``."""
    model = _Floordiv()
    compiled = lucid.compile(model)
    a, b = _operands()
    compiled(a, b)

    info = compiled.cache_info()
    assert info["entries"] == 1, (
        "integer // did not reach the graph — the op is most likely not "
        "recording its trace I/O again, which routes the whole model to "
        f"eager: {info}"
    )
    assert len(info["eager_only"]) == 0


def test_compiled_floordiv_floors_like_python() -> None:
    """Truncation and flooring differ only when the signs disagree."""
    model = _Floordiv()
    a, b = _operands()
    want = np.array([x // y for x, y in CASES], dtype=np.float32)

    eager = model(a, b).numpy()
    compiled = lucid.compile(model)(a, b).numpy()

    assert np.array_equal(eager, want), f"eager disagrees with Python: {eager}"
    assert np.array_equal(compiled, want), (
        f"compiled floordiv is truncating, not flooring: got {compiled.tolist()}, "
        f"want {want.tolist()}"
    )


def test_rope_style_index_split_compiles() -> None:
    """The shape this was found in: one flat id split into three axes."""

    class _Axes(nn.Module):
        def forward(self, ids: lucid.Tensor) -> lucid.Tensor:
            per_frame = 16 * 16
            depth = ids // per_frame
            rest = ids - depth * per_frame
            row = rest // 16
            col = rest - row * 16
            return lucid.stack([depth, row, col], dim=-1).to(lucid.float32)

    ids = lucid.arange(0, 512).to(lucid.int64).to(COMPILE_DEVICE)
    model = _Axes()
    compiled = lucid.compile(model)
    got = compiled(ids).numpy()

    assert compiled.cache_info()["entries"] == 1
    flat = np.arange(512)
    want = np.stack([flat // 256, (flat % 256) // 16, flat % 16], -1).astype(np.float32)
    assert np.array_equal(got, want)
