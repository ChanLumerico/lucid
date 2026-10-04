"""DataLoader worker processes: their random streams and their lifecycle.

Two families of defect lived here, and neither announced itself.

* Workers never seeded Lucid's generator, so every worker drew the same
  "random" augmentation and every epoch repeated the last one — a run
  trained on a fraction of the variation it was configured for, and
  converged anyway.
* The pool's bookkeeping: ``get_worker_info().num_workers`` was always 0,
  a ``break`` out of a persistent epoch let its leftovers leak into the
  next one (duplicates and gaps, right batch count), and a worker that
  raised or died left the loader waiting forever.

Every test here is bounded — an alarm turns a hang into a failure — and
checks that it left no worker process behind.  Datasets the workers
unpickle live in :mod:`._worker_datasets`.
"""

import collections
import gc
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import lucid
from lucid.utils.data import DataLoader

from lucid.test.unit.utils.data import _worker_datasets as wd

_WORKERS = 2
# Per test, generous: a worker spawn costs about a second.
_TEST_LIMIT_S = 120
_REPO = Path(__file__).resolve().parents[5]


@pytest.fixture(autouse=True)
def _bounded_and_tidy():
    """Fail rather than hang, and leave no worker process behind."""
    before = set(mp.active_children())

    def _hung(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"DataLoader test still running after {_TEST_LIMIT_S}s")

    previous = signal.signal(signal.SIGALRM, _hung)
    signal.alarm(_TEST_LIMIT_S)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    gc.collect()
    deadline = time.monotonic() + 15.0
    while set(mp.active_children()) - before and time.monotonic() < deadline:
        time.sleep(0.05)
    left = set(mp.active_children()) - before
    assert not left, f"worker processes left behind: {left}"


def _loader(dataset, **kwargs):
    # A stuck worker fails the test through ``timeout`` before the alarm.
    kwargs.setdefault("timeout", 60.0)
    return DataLoader(dataset, num_workers=_WORKERS, **kwargs)


def _draws(loader):
    """``[(lucid_draw, python_draw, worker_id), ...]`` for one epoch."""
    return [(float(a), float(b), int(w)) for a, b, w in loader]


def _by_worker(samples, column):
    out = collections.defaultdict(list)
    for sample in samples:
        out[sample[2]].append(sample[column])
    return out


# ── CHA-180: every worker, every epoch, its own stream ───────────────────────


def test_workers_draw_different_values_and_epochs_draw_new_ones():
    """Both workers drew ``0.0194`` and both drew ``0.7874``, and the next
    epoch drew exactly those again."""
    lucid.manual_seed(0)
    loader = _loader(wd.Draws(8), batch_size=None, collate_fn=wd.as_is)
    first, second = _draws(loader), _draws(loader)
    for column in (0, 1):  # Lucid's generator, Python's ``random``
        workers = _by_worker(first, column)
        assert sorted(workers) == [0, 1]
        assert not set(workers[0]) & set(workers[1])
        assert not {s[column] for s in first} & {s[column] for s in second}


def test_persistent_workers_draw_new_values_each_epoch():
    """A persistent pool is seeded once; its streams must carry on rather
    than restart, or every epoch repeats the first one's augmentations."""
    lucid.manual_seed(0)
    loader = _loader(
        wd.Draws(6), batch_size=None, collate_fn=wd.as_is, persistent_workers=True
    )
    first, second = _draws(loader), _draws(loader)
    assert not {s[0] for s in first} & {s[0] for s in second}
    workers = _by_worker(second, 0)
    assert not set(workers[0]) & set(workers[1])
    del loader


def test_manual_seed_reproduces_what_the_workers_draw():
    def run():
        loader = _loader(wd.Draws(6), batch_size=None, collate_fn=wd.as_is)
        return _draws(loader)

    lucid.manual_seed(3)
    first = run()
    lucid.manual_seed(3)
    assert run() == first
    lucid.manual_seed(4)
    assert run() != first


def test_a_generator_reproduces_the_workers_and_isolates_them():
    """``generator=`` decides the shuffle *and* the worker seeds, whatever
    the global generator does meanwhile."""

    def run(seed):
        loader = _loader(
            wd.Draws(6),
            batch_size=None,
            collate_fn=wd.as_is,
            shuffle=True,
            generator=lucid.Generator(seed),
        )
        return _draws(loader)

    lucid.manual_seed(1)
    first = run(7)
    lucid.manual_seed(2)
    assert run(7) == first
    assert run(8) != first


def test_the_shuffle_does_not_depend_on_num_workers():
    lucid.manual_seed(5)
    single = [int(i) for i in DataLoader(wd.Indices(10), batch_size=None, shuffle=True)]
    lucid.manual_seed(5)
    pooled = [int(i) for i in _loader(wd.Indices(10), batch_size=None, shuffle=True)]
    assert pooled == single
    assert sorted(single) == list(range(10))


# ── CHA-181: what a worker is told about itself ──────────────────────────────


def test_worker_info_describes_the_pool_and_runs_before_the_first_fetch():
    """``num_workers`` read ``index_queue.maxsize``, which an ``mp.Queue``
    does not have, so it was always 0 — and the module's own sharding
    example, ``range(info.id, n, info.num_workers)``, failed on step 0."""
    samples = list(
        _loader(
            wd.Info(6),
            batch_size=None,
            collate_fn=wd.as_is,
            worker_init_fn=wd.offset_by_worker,
        )
    )
    seeds = {}
    for worker_id, num_workers, seed, offset in samples:
        assert num_workers == _WORKERS
        assert offset == 1000 + worker_id
        seeds[worker_id] = seed
    assert sorted(seeds) == [0, 1]
    assert seeds[1] - seeds[0] == 1  # base_seed + worker_id


def test_sharding_an_iterable_dataset_through_worker_info_covers_it_once():
    batches = list(_loader(wd.Stream(10, shard=True), batch_size=2))
    assert sorted(int(i) for batch in batches for i in batch) == list(range(10))


def test_an_unsharded_iterable_dataset_is_read_by_every_worker():
    """``num_workers`` used to be ignored for an iterable dataset, which ran
    in the main process.  Each worker iterates its own copy."""
    counts = collections.Counter(
        int(i) for i in _loader(wd.Stream(5, shard=False), batch_size=None)
    )
    assert counts == {i: _WORKERS for i in range(5)}


def test_drop_last_applies_to_each_worker_s_share_of_a_stream():
    batches = list(_loader(wd.Stream(10, shard=True), batch_size=4, drop_last=True))
    assert sorted(len(b) for b in batches) == [4, 4]


def test_a_persistent_iterable_loader_restarts_the_stream_each_epoch():
    loader = _loader(wd.Stream(6, shard=True), batch_size=2, persistent_workers=True)
    for _ in range(2):
        seen = sorted(int(i) for batch in loader for i in batch)
        assert seen == list(range(6))
    del loader


# ── CHA-181: epochs on a persistent pool ─────────────────────────────────────


@pytest.mark.parametrize("persistent", [True, False])
def test_breaking_out_of_an_epoch_does_not_leak_into_the_next(persistent):
    """After a ``break`` the next epoch gave ``[3,0,1,5,9,11,2,7,1,11,6,10]``
    — 1 and 11 twice, 4 and 8 never — because the abandoned epoch's
    batches were still queued under the same sequence numbers."""
    lucid.manual_seed(0)
    loader = _loader(
        wd.Indices(12), batch_size=1, shuffle=True, persistent_workers=persistent
    )
    for _ in range(2):
        for i, _batch in enumerate(loader):
            if i == 2:
                break
        epoch = [int(b) for b in loader]
        assert sorted(epoch) == list(range(12)), epoch
    del loader


# ── CHA-181: a worker that fails ─────────────────────────────────────────────


def test_a_worker_exception_is_reraised_with_the_worker_s_traceback():
    loader = _loader(wd.Raises(6, bad=3), batch_size=1)
    with pytest.raises(KeyError) as caught:
        list(loader)
    message = str(caught.value)
    assert "DataLoader worker process" in message
    assert "Original Traceback" in message
    assert "no sample 3" in message


def test_the_pool_survives_a_worker_exception():
    """The next epoch hung forever: the exception shut the workers down but
    the persistent loader kept feeding their queues.  It now carries on —
    within the epoch (a caller may skip the bad batch) and after it."""
    loader = _loader(wd.Raises(6, bad=3), batch_size=1, persistent_workers=True)
    for _ in range(2):
        it = iter(loader)
        seen, failures = [], 0
        while True:
            try:
                seen.append(int(next(it)))
            except KeyError:
                failures += 1
            except StopIteration:
                break
        assert (seen, failures) == ([0, 1, 2, 4, 5], 1)
    del it, loader


def test_a_worker_init_fn_error_is_reraised():
    loader = _loader(wd.Indices(4), batch_size=1, worker_init_fn=wd.failing_init)
    with pytest.raises(ValueError, match="init failed"):
        list(loader)


def test_a_batch_that_cannot_be_sent_raises_instead_of_hanging():
    """The result queue pickles in a background thread; a batch that does
    not pickle was printed and dropped, and the loader waited for it."""
    loader = _loader(wd.Unpicklable(), batch_size=None, collate_fn=wd.as_is)
    with pytest.raises(Exception, match="while sending a batch"):
        list(loader)


def test_a_worker_that_dies_raises_instead_of_hanging():
    loader = _loader(wd.Dies(6, bad=2), batch_size=1, timeout=0.0)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        list(loader)


def test_a_slow_worker_times_out():
    loader = _loader(wd.Slow(4, delay=1.5), batch_size=1, timeout=0.2)
    with pytest.raises(RuntimeError, match="timed out"):
        list(loader)


def test_a_persistent_loader_starts_a_new_pool_after_a_shutdown():
    """A timeout shuts the pool down; the next epoch must not feed it."""
    dataset = wd.Slow(4, delay=1.5)
    loader = _loader(dataset, batch_size=1, timeout=0.2, persistent_workers=True)
    with pytest.raises(RuntimeError, match="timed out"):
        list(loader)
    dataset.delay = 0.0  # the new pool unpickles the dataset afresh
    loader.timeout = 60.0  # and needs a moment to start
    assert [int(b) for b in loader] == [0, 1, 2, 3]
    del loader


def test_workers_exit_when_the_main_process_is_killed():
    """An orphaned worker would otherwise wait on its index queue forever."""
    script = textwrap.dedent("""
        import os, signal
        from lucid.utils.data import DataLoader
        from lucid.test.unit.utils.data import _worker_datasets as wd

        loader = DataLoader(
            wd.Pids(), batch_size=None, num_workers=2, collate_fn=wd.as_is,
            timeout=60.0, persistent_workers=True,
        )
        print(" ".join(str(p) for p in sorted(set(loader))), flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
        """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    # The persistent pool is still up when its parent is killed.
    pool = [int(p) for p in proc.stdout.split()]
    assert len(pool) == _WORKERS

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    deadline = time.monotonic() + 20.0
    while any(alive(p) for p in pool) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(alive(p) for p in pool)
