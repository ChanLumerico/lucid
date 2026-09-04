"""A compiled model must answer its input, not the one it was traced with.

An op that never records its trace I/O leaves the node with the empty
input list ``on_op_enter`` seeded, and its result never becomes a traced
tensor. The builder then drops the node as dead — correctly, by its own
rule — and the *consumer* meets the result as a fresh external feed,
bound once, at trace time. The compiled model reports success, agrees
with eager on the first call, and returns that same answer forever.

That is the worst shape a defect can take here: no crash, no fallback,
no warning. ``fft.fftn`` was off by 20 on the second call.

The test is the property rather than the mechanism: compile, call with
two inputs that eager distinguishes, and require the output to move.
Whether an op compiles or falls back is not the point — either is
honest. Answering the previous question is not.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

# Every one of these used to freeze, except the last two, which are
# controls: one op that compiles and one that falls back, both of which
# have to keep tracking their input.
CASES = [
    ("fftn", lambda t: lucid.real(lucid.fft.fftn(t)), (2, 4)),
    ("ifftn", lambda t: lucid.real(lucid.fft.ifftn(lucid.complex(t, t))), (2, 4)),
    ("rfftn", lambda t: lucid.real(lucid.fft.rfftn(t)), (2, 4)),
    ("irfftn", lambda t: lucid.fft.irfftn(lucid.complex(t, t)), (2, 4)),
    ("nonzero", lambda t: lucid.nonzero(t > 0)[0], (8,)),
    ("histogram2d", lambda t: lucid.histogram2d(t[:, 0], t[:, 1], bins=4)[0], (16, 2)),
    ("histogramdd", lambda t: lucid.histogramdd(t, bins=[4, 4])[0], (16, 2)),
    ("relu-control", lambda t: t.relu(), (2, 4)),
    ("svd-control", lambda t: lucid.linalg.svd(t)[1], (4, 4)),
]


class _Apply(nn.Module):
    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)  # type: ignore[operator]


@pytest.mark.parametrize(("name", "fn", "shape"), CASES, ids=[c[0] for c in CASES])
def test_the_second_call_answers_the_second_input(name, fn, shape):
    lucid.manual_seed(0)
    model = _Apply(fn).eval()
    first = lucid.randn(*shape)
    second = lucid.randn(*shape) * 7 + 3

    want_first = model(first)
    want_second = model(second)
    spread = float((want_first - want_second).abs().max().item())
    assert spread > 1e-6, "the probe itself is blind — eager gives the same answer"

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got_first = compiled(first.to("metal")).to("cpu")
    got_second = compiled(second.to("metal")).to("cpu")

    moved = float((got_first - got_second).abs().max().item())
    assert moved > 1e-6, (
        f"{name}: the compiled model returned the same values for two different "
        "inputs — its result was frozen into the graph at trace time"
    )
    scale = max(float(want_second.abs().max().item()), 1e-30)
    assert float((got_second - want_second).abs().max().item()) / scale < 1e-5


def test_a_pure_python_composite_that_reads_values_is_still_frozen():
    """``lucid.histogram`` (1-D) is a Python composite over ``.item()``.

    Reading an element during tracing takes the trace-time value, so the
    whole histogram is computed on the host and baked in as constants —
    nothing in the graph depends on the input. This is not the tracer
    gap the tests above cover; it is what any composite that reads
    values does, and no amount of trace wiring changes it.

    The test records the behaviour rather than blessing it, so that a
    fix registers as a failure here instead of going unnoticed.
    """
    lucid.manual_seed(0)
    model = _Apply(lambda t: lucid.histogram(t, bins=4)[0]).eval()
    first = lucid.randn(16)
    second = lucid.randn(16) * 7 + 3
    assert float((model(first) - model(second)).abs().max().item()) > 1e-6

    _C_engine.compile.session_cache_clear()
    compiled = lucid.compile.compile(model.to("metal"))
    got_first = compiled(first.to("metal")).to("cpu")
    got_second = compiled(second.to("metal")).to("cpu")
    assert float((got_first - got_second).abs().max().item()) <= 1e-6
