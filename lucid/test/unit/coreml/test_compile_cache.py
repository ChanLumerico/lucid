"""A package is compiled once, and every load after that opens the same model.

Core ML specialises a loaded model into a bundle in its own cache — for a
GPU segment a full copy of the weights, 1.1 GB for YOLO v1 — keyed by the
compiled model's path and file identity.  ``lucid.coreml`` used to compile
on every load, into a fresh temporary path, so every load wrote a bundle
nothing would read again; ``load`` opened the package twice besides.  A
development machine's cache reached 121 GB (CHA-24).

Now the compiled model is kept in ``LUCID_COREML_CACHE_DIR``, one per
package content, and these tests hold the cache to what it promises: the
same path and no new bundle on a reload, a new entry for new content, the
limit, the concurrency, and an error instead of Core ML ending the process
when the disk is short.
"""

import errno
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.coreml import _cache
from lucid.test.unit.coreml._helpers import bundles_under

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Conv2d(3, 8, 3, padding=1)
        self.head = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.body(x).relu().mean(dim=(2, 3)))


_X_SHAPE = (1, 3, 16, 16)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cache of this test's own, so what it counts is what it made."""
    root = tmp_path / "cache"
    monkeypatch.setenv("LUCID_COREML_CACHE_DIR", str(root))
    monkeypatch.delenv("LUCID_COREML_CACHE_LIMIT", raising=False)
    return root


def _write(path: Path, seed: int = 0) -> str:
    """Export a small model to ``path`` and close the handle."""
    lucid.manual_seed(seed)
    cml.export(_Small().eval(), lucid.randn(*_X_SHAPE), str(path)).close()
    return str(path)


def _entries(root: Path) -> list[Path]:
    return sorted(root.glob("*.mlmodelc")) if root.is_dir() else []


class _Counting:
    """Wraps an engine call and counts it."""

    def __init__(self, inner: Callable[..., object]) -> None:
        self.inner = inner
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, *args: object, **kwargs: object) -> object:
        with self._lock:
            self.calls += 1
        return self.inner(*args, **kwargs)


class TestAReloadOpensWhatTheFirstLoadCompiled:
    def test_the_same_compiled_model_and_no_new_bundle(
        self, tmp_path: Path, cache: Path, core_ml_home: Path | None
    ) -> None:
        before = bundles_under(core_ml_home) if core_ml_home is not None else set()
        path = _write(tmp_path / "m.mlpackage")
        first_load = (
            bundles_under(core_ml_home) - before if core_ml_home is not None else set()
        )

        opened = []
        for _ in range(3):
            with cml.load(path) as reopened:
                opened.append(reopened._lease.path)
                assert tuple(reopened.predict(lucid.randn(*_X_SHAPE)).shape) == (1, 4)

        assert len(set(opened)) == 1
        assert _entries(cache) == [Path(opened[0])]
        if core_ml_home is None or not first_load:
            pytest.skip("Core ML keeps its bundles outside this session's home")
        # The export's own load wrote one load's worth; three more wrote none.
        assert bundles_under(core_ml_home) - before == first_load

    def test_load_opens_the_package_once(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write(tmp_path / "m.mlpackage")
        loads = _Counting(_C_engine.coreml.load_model)
        monkeypatch.setattr(_C_engine.coreml, "load_model", loads)
        with cml.load(path):
            pass
        assert loads.calls == 1

    def test_neither_a_reload_nor_a_compute_plan_compiles(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write(tmp_path / "m.mlpackage")
        compiles = _Counting(_C_engine.coreml.compile_model)
        monkeypatch.setattr(_C_engine.coreml, "compile_model", compiles)
        with cml.load(path) as reopened:
            assert reopened.compute_plan().total_compute > 0
        assert compiles.calls == 0

    def test_the_kept_model_outlives_the_handle(
        self, tmp_path: Path, cache: Path
    ) -> None:
        path = _write(tmp_path / "m.mlpackage")
        (entry,) = _entries(cache)
        assert (entry / "coremldata.bin").is_file()
        with cml.load(path):
            pass
        assert (entry / "coremldata.bin").is_file()


class TestTheKeyIsTheContent:
    def test_the_same_bytes_elsewhere_are_the_same_entry(
        self, tmp_path: Path, cache: Path
    ) -> None:
        path = _write(tmp_path / "m.mlpackage")
        moved = shutil.copytree(path, tmp_path / "copy.mlpackage")
        with cml.load(path) as one, cml.load(str(moved)) as other:
            assert one._lease.path == other._lease.path
        assert len(_entries(cache)) == 1

    def test_new_weights_at_the_same_path_are_a_new_entry(
        self, tmp_path: Path, cache: Path
    ) -> None:
        path = tmp_path / "m.mlpackage"
        _write(path, seed=0)
        with cml.load(str(path)) as old:
            before = old._lease.path
        _write(path, seed=1)
        with cml.load(str(path)) as new:
            after = new._lease.path
        assert before != after
        assert len(_entries(cache)) == 2

    def test_the_os_and_core_ml_builds_are_part_of_it(self) -> None:
        fingerprint = _cache._fingerprint()
        assert "coreml=" in fingerprint and "os=" in fingerprint
        assert "coreml=unknown" not in fingerprint


class TestTheCacheKeepsToItsLimit:
    def _two_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache: Path
    ) -> tuple[list[str], int]:
        paths = [_write(tmp_path / f"m{i}.mlpackage", seed=i) for i in range(2)]
        sizes = [_cache._tree_bytes(str(entry)) for entry in _entries(cache)]
        # Room for two, not three.
        monkeypatch.setenv("LUCID_COREML_CACHE_LIMIT", str(sum(sizes) + 16))
        return paths, max(sizes)

    def test_the_least_recently_used_goes_first(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (first, second), _size = self._two_kept(tmp_path, monkeypatch, cache)
        with cml.load(first) as handle:
            kept = handle._lease.path
        with cml.load(second) as handle:
            dropped = handle._lease.path
        # Age both, then open the first again: a load is what marks an
        # entry used, whatever order they were written in.
        for entry in (kept, dropped):
            os.utime(Path(entry).with_suffix(".json"), (1.0, 1.0))
        with cml.load(first):
            pass

        third = _write(tmp_path / "m2.mlpackage", seed=2)

        left = {str(entry) for entry in _entries(cache)}
        assert kept in left and dropped not in left
        assert len(left) == 2
        assert not Path(dropped).with_suffix(".lock").exists()
        assert not Path(dropped).with_suffix(".json").exists()
        with cml.load(third):
            pass

    def test_an_open_handle_keeps_its_model(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (first, second), _size = self._two_kept(tmp_path, monkeypatch, cache)
        monkeypatch.setenv("LUCID_COREML_CACHE_LIMIT", "1")  # room for nothing
        with cml.load(first) as held:
            _write(tmp_path / "m2.mlpackage", seed=2)
            left = {str(entry) for entry in _entries(cache)}
            assert held._lease.path in left
            assert tuple(held.predict(lucid.randn(*_X_SHAPE)).shape) == (1, 4)
            # Of the three only the held one and the newest survive.
            assert len(left) == 2

    def test_another_process_holding_a_model_keeps_it(
        self, tmp_path: Path, cache: Path
    ) -> None:
        path = _write(tmp_path / "m.mlpackage")
        (entry,) = _entries(cache)
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import fcntl, os, sys\n"
                "fd = os.open(sys.argv[1], os.O_RDWR)\n"
                "fcntl.flock(fd, fcntl.LOCK_SH)\n"
                "print('held', flush=True)\n"
                "sys.stdin.read()\n",
                str(entry.with_suffix(".lock")),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None and holder.stdout.readline() == "held\n"
            assert cml.empty_cache() == 0
            assert entry.is_dir()
        finally:
            assert holder.stdin is not None
            holder.stdin.close()
            holder.wait(timeout=30)
        assert cml.empty_cache() > 0
        assert not entry.exists()
        with cml.load(path):  # and is compiled again on the next load
            pass
        assert len(_entries(cache)) == 1

    def test_a_model_larger_than_the_limit_is_still_kept_once(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LUCID_COREML_CACHE_LIMIT", "1")
        path = _write(tmp_path / "m.mlpackage")
        assert len(_entries(cache)) == 1
        compiles = _Counting(_C_engine.coreml.compile_model)
        monkeypatch.setattr(_C_engine.coreml, "compile_model", compiles)
        with cml.load(path):
            pass
        assert compiles.calls == 0

    def test_zero_turns_it_off(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LUCID_COREML_CACHE_LIMIT", "0")
        path = _write(tmp_path / "m.mlpackage")
        handle = cml.load(path)
        private = handle._lease.path
        assert Path(private).is_dir()
        assert not Path(private).is_relative_to(cache)
        handle.close()
        assert not Path(private).exists()
        assert _entries(cache) == []

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("4G", 4 << 30),
            ("4GiB", 4 << 30),
            ("1.5M", 3 << 19),
            ("512kb", 512 << 10),
            ("1000", 1000),
            ("0", 0),
        ],
    )
    def test_a_limit_reads_as_a_size(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
    ) -> None:
        monkeypatch.setenv("LUCID_COREML_CACHE_LIMIT", raw)
        assert _cache.cache_limit() == expected

    @pytest.mark.parametrize("raw", ["lots", "-1", "nan", "inf", "4X"])
    def test_a_limit_that_is_not_a_size_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("LUCID_COREML_CACHE_LIMIT", raw)
        with pytest.raises(ValueError, match="LUCID_COREML_CACHE_LIMIT"):
            _cache.cache_limit()


class TestWhereItLives:
    def test_the_default_is_beside_the_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LUCID_COREML_CACHE_DIR", raising=False)
        monkeypatch.delenv("LUCID_HOME", raising=False)
        assert _cache.cache_dir() == os.path.expanduser("~/.cache/lucid/coreml")

    def test_lucid_home_moves_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("LUCID_COREML_CACHE_DIR", raising=False)
        monkeypatch.setenv("LUCID_HOME", str(tmp_path))
        assert _cache.cache_dir() == str(tmp_path / "coreml")

    def test_an_unwritable_place_compiles_per_load_and_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        blocked = tmp_path / "a-file"
        blocked.write_text("")
        monkeypatch.setenv("LUCID_COREML_CACHE_DIR", str(blocked / "cache"))
        lucid.manual_seed(0)
        with pytest.warns(RuntimeWarning, match="cannot keep compiled models"):
            handle = cml.export(
                _Small().eval(), lucid.randn(*_X_SHAPE), str(tmp_path / "m.mlpackage")
            )
        private = handle._lease.path
        handle.close()
        assert not Path(private).exists()


class TestTwoLoadsAtOnce:
    def test_two_threads_compile_a_new_package_once(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write(tmp_path / "m.mlpackage")
        assert cml.empty_cache() > 0
        compiles = _Counting(_C_engine.coreml.compile_model)
        monkeypatch.setattr(_C_engine.coreml, "compile_model", compiles)

        start = threading.Barrier(2)
        opened: list[str] = []
        failed: list[BaseException] = []

        def load() -> None:
            try:
                start.wait()
                with cml.load(path) as handle:
                    handle.predict(lucid.randn(*_X_SHAPE))
                    opened.append(handle._lease.path)
            except BaseException as exc:  # noqa: BLE001 - reported below
                failed.append(exc)

        threads = [threading.Thread(target=load) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not failed, failed
        assert len(opened) == 2 and opened[0] == opened[1]
        assert compiles.calls == 1
        assert len(_entries(cache)) == 1
        assert not list(cache.glob(".staging-*"))

    def test_a_process_that_loses_the_race_opens_the_winner(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another process renames its copy onto the entry mid-compile."""
        path = _write(tmp_path / "m.mlpackage")
        assert cml.empty_cache() > 0
        real = _C_engine.coreml.compile_model
        key, _size = _cache._identify(path)
        entry = cache / f"{key}.mlmodelc"

        def compile_while_another_wins(package: str) -> str:
            shutil.move(real(package), entry)
            return real(package)

        monkeypatch.setattr(
            _C_engine.coreml, "compile_model", compile_while_another_wins
        )
        with cml.load(path) as handle:
            assert handle._lease.path == str(entry)
            assert tuple(handle.predict(lucid.randn(*_X_SHAPE)).shape) == (1, 4)
        assert _entries(cache) == [entry]
        assert not list(cache.glob(".staging-*"))


class TestADiskTooFullForCoreML:
    """Core ML ends the process when it runs out of disk part way.

    ``LLVM ERROR: IO failure on output stream: No space left on device``,
    then ``exit(1)`` — reproduced on a small disk image. A load checks
    first and raises what the caller can catch.
    """

    @pytest.fixture
    def full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_cache, "_free_bytes", lambda path: 64 << 20)

    def test_a_new_package_is_refused_before_compiling(
        self,
        tmp_path: Path,
        cache: Path,
        monkeypatch: pytest.MonkeyPatch,
        full: None,
    ) -> None:
        compiles = _Counting(_C_engine.coreml.compile_model)
        monkeypatch.setattr(_C_engine.coreml, "compile_model", compiles)
        lucid.manual_seed(0)
        with pytest.raises(OSError) as raised:
            cml.export(
                _Small().eval(), lucid.randn(*_X_SHAPE), str(tmp_path / "m.mlpackage")
            )
        assert raised.value.errno == errno.ENOSPC
        message = str(raised.value)
        assert "lucid.coreml" in message and "m.mlpackage" in message
        assert "free" in message and "LUCID_COREML_CACHE_DIR" in message
        assert compiles.calls == 0
        assert _entries(cache) == []
        # The package itself was written; only loading it was refused.
        assert (tmp_path / "m.mlpackage").is_dir()

    def test_a_cached_package_is_refused_before_loading(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write(tmp_path / "m.mlpackage")
        loads = _Counting(_C_engine.coreml.load_model)
        monkeypatch.setattr(_C_engine.coreml, "load_model", loads)
        monkeypatch.setattr(_cache, "_free_bytes", lambda where: 64 << 20)
        with pytest.raises(OSError) as raised:
            cml.load(path)
        assert raised.value.errno == errno.ENOSPC
        assert "bundle Core ML specialises" in str(raised.value)
        assert loads.calls == 0
        # The refusal let go of the entry it looked up.
        assert cml.empty_cache() > 0

    def test_room_enough_loads(
        self, tmp_path: Path, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_cache, "_free_bytes", lambda where: 1 << 40)
        with cml.load(_write(tmp_path / "m.mlpackage")) as handle:
            assert tuple(handle.predict(lucid.randn(*_X_SHAPE)).shape) == (1, 4)
