"""Waiting for the GPU does not hold the GIL.

``lucid.eval``, readbacks and compiled runs waited for the GPU while holding
the GIL, so every other Python thread in the process stalled for the whole
pass — a server thread froze for each 1.6-3 s model call (reported while
porting Self-Forcing).  The waits release it now.
"""

import threading
import time

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
@pytest.mark.parametrize(
    "wait",
    [lambda t: lucid.eval(t), lambda t: t.sum().item(), lambda t: t.tolist()],
    ids=["eval", "item", "tolist"],
)
def test_another_thread_runs_while_the_gpu_works(wait) -> None:  # type: ignore[no-untyped-def]
    a = lucid.randn(1024, 1024, device="metal")
    b = a
    for _ in range(40):
        b = (b @ a) / 32
    ticks = [0]
    stop = threading.Event()

    def spin() -> None:
        while not stop.is_set():
            ticks[0] += 1

    thread = threading.Thread(target=spin)
    thread.start()
    time.sleep(0.02)
    before = ticks[0]
    start = time.perf_counter()
    wait(b)
    elapsed = time.perf_counter() - start
    during = ticks[0] - before
    stop.set()
    thread.join()
    # A held GIL lets the spinner run only at the interpreter's switch
    # interval; released, it runs throughout.  Judge by rate, not count.
    assert elapsed > 0.0
    assert during / elapsed > 10_000, (during, elapsed)
