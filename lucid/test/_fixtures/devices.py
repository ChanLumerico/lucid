"""Device fixtures — drives CPU/Metal cross-validation.

The default ``device`` fixture parametrizes every requesting test over
the available compute streams.  Metal is detected by attempting a
zero-cost allocation and skipped at parametrize time when unavailable.
"""

import functools
from collections.abc import Iterator

import pytest


@functools.lru_cache(maxsize=1)
def metal_available() -> bool:
    """Return True if Apple Metal is usable — try a tiny allocation
    and let any failure bubble up as ``False``."""
    try:
        from lucid._C import engine as _C_engine

        _ = _C_engine.zeros([1], _C_engine.F32, _C_engine.GPU)
    except Exception:
        return False
    return True


def _device_params() -> list[str]:
    return ["cpu", "metal"] if metal_available() else ["cpu"]


@pytest.fixture(params=_device_params())
def device(request: pytest.FixtureRequest) -> str:
    """Yield each available compute device.  Tests using this fixture
    automatically run on CPU and (when present) Metal."""
    return str(request.param)


@pytest.fixture
def device_cpu_only() -> str:
    return "cpu"


@pytest.fixture
def device_gpu_only() -> str:
    if not metal_available():
        pytest.skip("Metal device not available on this host")
    return "metal"


@pytest.fixture
def cross_device_pair() -> Iterator[tuple[str, str]]:
    """Yield ``("cpu", "metal")`` once when both are present; otherwise
    skip.  Intended for "device drift" tests that compare CPU and GPU
    outputs of the same op."""
    if not metal_available():
        pytest.skip("CPU↔Metal cross-device tests need Metal")
    yield ("cpu", "metal")
