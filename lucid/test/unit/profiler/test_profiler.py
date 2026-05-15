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
