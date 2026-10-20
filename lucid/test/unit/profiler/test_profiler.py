"""``lucid.profiler`` — timing / memory hook smoke."""

import pytest

import lucid


class TestProfilerSurface:
    def test_present(self) -> None:
        assert hasattr(lucid, "profiler")


class TestProfilerContext:
    def test_basic(self) -> None:
        if not hasattr(lucid.profiler, "Profiler"):
            pytest.skip("Profiler class not exposed")
        with lucid.profiler.Profiler() as p:
            _ = lucid.zeros(3, 4) + lucid.ones(3, 4)
        # Just check the context manager exits cleanly.
        assert p is not None
        assert p.memory_stats() is not None

    def test_with_memory_false_omits_memory_stats(self) -> None:
        with lucid.profiler.profile(with_memory=False) as p:
            _ = lucid.zeros(2, 2) + lucid.ones(2, 2)
        assert p.memory_stats() is None
