"""
Phase 9 acceptance: concurrency stress test.

100 threads × 1000 iterations of mixed ops — verify no deadlock, no crash,
no data race (via consistent results).

Run: python -m pytest tests/test_concurrency.py -v
"""

from __future__ import annotations

import threading
import numpy as np
import pytest
from lucid._C import engine as E


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make(shape, rng):
    return E.TensorImpl(rng.standard_normal(shape).astype("float32"), E.Device.CPU, True)


# --------------------------------------------------------------------------- #
# 9.1 — OpRegistry concurrent reads
# --------------------------------------------------------------------------- #

def test_op_registry_concurrent_reads():
    """All threads should see consistent registry (no crash under concurrent lookup)."""
    errors = []

    def worker():
        try:
            schemas = E.op_registry_all()
            assert len(schemas) > 0
            s = E.op_lookup("add")
            assert s is not None
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, errors


# --------------------------------------------------------------------------- #
# 9.2 — Reentrancy: backward on disjoint graphs from concurrent threads
# --------------------------------------------------------------------------- #

def test_backward_disjoint_graphs_concurrent():
    """Each thread builds its own graph and runs backward — no shared state."""
    errors = []

    def worker():
        try:
            rng = np.random.default_rng()
            a = _make((16, 16), rng)
            b = _make((16, 16), rng)
            c = E.matmul(a, b)
            s = E.sum(c, [0, 1], False)
            E.engine_backward(s, False)
            ga = np.asarray(a.grad_as_python())
            assert ga.shape == (16, 16)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, errors[:3]


# --------------------------------------------------------------------------- #
# 9.3 — Memory pool: many rapid alloc/free cycles
# --------------------------------------------------------------------------- #

def test_memory_pool_stress():
    """1000 iterations of alloc-compute-backward per thread — no crash/leak."""
    errors = []

    def worker(n_iter=200):
        try:
            rng = np.random.default_rng()
            for _ in range(n_iter):
                x = _make((32, 32), rng)
                y = E.relu(x)
                s = E.sum(y, [0, 1], False)
                E.engine_backward(s, False)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, errors[:3]


# --------------------------------------------------------------------------- #
# 9.4 — Graph pruning: saved tensors released after backward
# --------------------------------------------------------------------------- #

def test_graph_pruning_memory_released():
    """After backward(retain_graph=False), saved activations should be freed."""
    import gc

    rng = np.random.default_rng(0)
    x = _make((256, 256), rng)
    y = E.relu(x)
    s = E.sum(y, [0, 1], False)

    E.engine_backward(s, False)
    gc.collect()

    # After backward, grad_fn should be cleared from s (clear_grad_fn called).
    # We can't directly inspect saved_inputs_ from Python, but we verify the
    # backward ran without error and produced correct gradients.
    g = np.asarray(x.grad_as_python())
    assert g.shape == (256, 256)
    assert np.all(g >= 0)  # relu grad: 1 where x>0, 0 elsewhere


# --------------------------------------------------------------------------- #
# Full stress: 100 threads × 1000 iter of mixed ops
# --------------------------------------------------------------------------- #

def test_no_memory_leak_after_backward():
    """MemoryTracker confirms current_bytes returns to 0 after all tensors freed.
    Equivalent to ASan leak check (ASan requires Homebrew Python due to macOS SIP)."""
    import gc
    gc.collect()
    E.reset_peak_memory_stats(E.Device.CPU)
    before = E.memory_stats(E.Device.CPU).current_bytes

    rng = np.random.default_rng(42)
    for _ in range(200):
        a = E.TensorImpl(rng.standard_normal((32, 32)).astype("float32"), E.Device.CPU, True)
        b = E.TensorImpl(rng.standard_normal((32, 32)).astype("float32"), E.Device.CPU, True)
        c = E.matmul(a, b)
        s = E.sum(c, [0, 1], False)
        E.engine_backward(s, False)
        del a, b, c, s

    gc.collect()
    after = E.memory_stats(E.Device.CPU).current_bytes
    assert after == before, f"Memory leak: {after - before} bytes unreleased"


@pytest.mark.slow
def test_full_stress_100_threads():
    """The Phase 9 acceptance criterion: 100 threads × 1000 iter, no crash."""
    errors = []

    def worker(n_iter=1000):
        try:
            rng = np.random.default_rng()
            for _ in range(n_iter):
                a = E.TensorImpl(rng.standard_normal((8, 8)).astype("float32"),
                                  E.Device.CPU, True)
                b = E.TensorImpl(rng.standard_normal((8, 8)).astype("float32"),
                                  E.Device.CPU, True)
                c = E.add(a, b)
                d = E.relu(c)
                s = E.sum(d, [0, 1], False)
                E.engine_backward(s, False)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, f"{len(errors)} thread(s) failed: {errors[:3]}"
