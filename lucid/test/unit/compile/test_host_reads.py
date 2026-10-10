"""A host read of a traced value runs the call eagerly, whichever API made it.

``.item()``, ``.tolist()``, ``.numpy()``, ``bool(t)``, ``float(t)`` and
``int(t)`` hand a tensor's values to Python, where the recording cannot
follow what is done with them. Traced once, the value read becomes a
constant and every later call answers the first one's question — no
crash, no warning (LCD-337).

The tracer marks such a trace unsupported, and the compiled call runs
eagerly with a ``CompileFallbackWarning``. It used to see a read only of a
value an op had already consumed, so reading an *argument* before any op
touched it (``sum(x.tolist())`` first thing in forward) still froze; the
arguments are now declared to the tracer up front.

The property is checked, not the mechanism: two inputs eager tells apart,
and the compiled second answer has to be eager's.
"""

import warnings
from collections.abc import Iterator

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile._host_reads import (
    CONSTANT_READS,
    HOST_READS,
    Apply,
    inputs,
)


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.fixture(autouse=True)
def _fresh_notices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    _C_engine.compile.session_cache_clear()
    yield


@pytest.mark.parametrize("name", list(HOST_READS))
def test_a_host_read_runs_eagerly_and_follows_the_input(name: str) -> None:
    """Between the calls both the input and the weights change."""
    first, second = inputs()
    reference = Apply(HOST_READS[name]).eval()
    want_first = reference(first)
    reference.change_state()
    want_second = reference(second)
    assert float((want_first - want_second).abs().max().item()) > 1e-3

    model = Apply(HOST_READS[name]).eval().to(COMPILE_DEVICE)
    compiled = lucid.compile.compile(model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got_first = compiled(first.to(COMPILE_DEVICE)).to("cpu")
        model.change_state()
        got_second = compiled(second.to(COMPILE_DEVICE)).to("cpu")

    for got, want in ((got_first, want_first), (got_second, want_second)):
        assert float((got - want).abs().max().item()) <= 1e-4 * float(
            want.abs().max().item()
        )
    said = [
        str(w.message)
        for w in caught
        if issubclass(w.category, fallback.CompileFallbackWarning)
    ]
    assert said and "host" in said[0], said
    assert compiled.cache_info()["eager_only"]


@pytest.mark.parametrize("name", list(CONSTANT_READS))
def test_a_read_that_feeds_nothing_stays_compiled(name: str) -> None:
    model = Apply(CONSTANT_READS[name]).eval()
    first, second = inputs()
    compiled = lucid.compile.compile(model.to(COMPILE_DEVICE))
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        compiled(first.to(COMPILE_DEVICE))
        got = compiled(second.to(COMPILE_DEVICE)).to("cpu")
    assert float((got - second * 3).abs().max().item()) <= 1e-5
    assert not compiled.cache_info()["eager_only"]
