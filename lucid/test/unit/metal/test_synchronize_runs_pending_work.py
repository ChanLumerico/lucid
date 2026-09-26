"""``lucid.metal.synchronize`` runs the work it waits for.

Lucid's Metal path is lazy: an op records a graph node and returns, and
nothing is submitted until a value is read.  ``synchronize`` waited only
for work already submitted — none — and returned at once, so a timing
bracketed by it measured nothing (a matmul loop reported 219 TFLOPS;
reported while porting Self-Forcing, CHA-12).  It now evaluates every GPU
tensor whose work is still pending before waiting.
"""

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")


def test_pending_work_runs_at_synchronize() -> None:
    _C_engine.synchronize_gpu()  # drain anything earlier tests left queued
    a = lucid.randn(64, 64, device="metal")
    b = a
    for _ in range(5):
        b = b @ a
    assert _C_engine.synchronize_gpu() >= 1
    # Nothing is left waiting, and the values are readable as they were.
    assert _C_engine.synchronize_gpu() == 0
    assert b.shape == (64, 64)


def test_the_public_functions_route_through_it() -> None:
    a = lucid.randn(32, 32, device="metal")
    _ = a @ a
    lucid.metal.synchronize()
    assert _C_engine.synchronize_gpu() == 0
    _ = a @ a
    event = lucid.metal.MetalEvent(enable_timing=True)
    event.record()
    assert _C_engine.synchronize_gpu() == 0


def test_a_dropped_tensor_is_not_kept_alive_or_run() -> None:
    _C_engine.synchronize_gpu()
    a = lucid.randn(16, 16, device="metal")
    tmp = a @ a
    del tmp
    # Only ``a`` (a constant draw, possibly still lazy) may remain; the
    # dropped product must not be resurrected by the tracking.
    assert _C_engine.synchronize_gpu() <= 1
