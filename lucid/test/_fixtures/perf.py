"""Performance fixtures — built on top of ``pytest-benchmark``.

The ``bench`` fixture wraps the upstream ``benchmark`` fixture so tests
degrade to a no-op timing call when ``pytest-benchmark`` isn't
installed.  This keeps the rest of the suite green even in the
slimmest dev environments while making the perf tier first-class
when the dep is present.
"""

import functools
import importlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@functools.lru_cache(maxsize=1)
def _benchmark_available() -> bool:
    try:
        importlib.import_module("pytest_benchmark")
        return True
    except ImportError:
        return False


@pytest.fixture
def bench(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """Return a ``benchmark``-compatible callable.

    When ``pytest-benchmark`` is installed we forward to its
    ``benchmark`` fixture; otherwise we fall back to a tiny inline
    runner that just calls the function once and records elapsed time
    so the test still asserts behaviour without skipping the suite.
    """
    if _benchmark_available():
        return request.getfixturevalue("benchmark")

    def _fallback(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        # Stash the timing so tests can assert on it if they want.
        _fallback.last_elapsed = elapsed  # type: ignore[attr-defined]
        return out

    return _fallback


def load_thresholds(area_dir: Path) -> dict[str, float]:
    """Load golden-timing thresholds from ``<area_dir>/_thresholds.json``.

    Returns an empty dict when the file is missing — perf tests can
    then run without enforcing a regression threshold (useful for
    bootstrapping a new area)."""
    path = area_dir / "_thresholds.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def assert_no_regression(
    name: str,
    elapsed_s: float,
    thresholds: dict[str, float],
    *,
    tolerance: float = 0.25,
) -> None:
    """Fail when ``elapsed_s`` exceeds ``thresholds[name] * (1 + tolerance)``.

    Missing entries are treated as "no threshold" so new perf tests
    don't have to gate the suite until a baseline is recorded.
    """
    threshold = thresholds.get(name)
    if threshold is None:
        return
    limit = threshold * (1.0 + tolerance)
    if elapsed_s > limit:
        raise AssertionError(
            f"perf regression {name!r}: {elapsed_s * 1e3:.3f} ms "
            f"> {limit * 1e3:.3f} ms (threshold {threshold * 1e3:.3f} ms "
            f"+{int(tolerance * 100)}%)"
        )
