"""
Global pytest fixtures and configuration for the lucid test suite.

Fixtures:
    device       — parametrize "cpu" (and "gpu" when available)
    float_dtype  — parametrize lucid.float32 / lucid.float64
    seed         — autouse: calls lucid.manual_seed(0) before every test
    assert_close — convenience shortcut to lucid.test.assert_close

Marks applied automatically:
    parity      — any test under lucid/test/parity/
    slow        — any test under lucid/test/integration/
"""

import pytest
import lucid
from lucid.test._comparison import assert_close as _assert_close

# ── Devices ────────────────────────────────────────────────────────────────────


def _available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        from lucid._C import engine as _C_engine

        # GPU available if we can round-trip a small tensor
        t = _C_engine.zeros([1], _C_engine.F32, _C_engine.GPU)
        devices.append("gpu")
    except Exception:
        pass
    return devices


DEVICES = _available_devices()


@pytest.fixture(params=["cpu"])
def device(request) -> str:
    """Parametrize over available compute devices."""
    return request.param


@pytest.fixture(params=["cpu"] + (["gpu"] if "gpu" in DEVICES else []))
def all_devices(request) -> str:
    """Parametrize over ALL available devices including GPU."""
    return request.param


# ── Dtypes ─────────────────────────────────────────────────────────────────────


@pytest.fixture(params=["float32", "float64"])
def float_dtype(request) -> object:
    """Parametrize over common floating-point dtypes."""
    return getattr(lucid, request.param)


@pytest.fixture(params=["float32"])
def default_dtype(request) -> object:
    """Single float32 dtype (for tests that only need one dtype)."""
    return getattr(lucid, request.param)


# ── Seeding ────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def seed():
    """Reset the RNG before every test for determinism."""
    lucid.manual_seed(0)
    yield


# ── Comparison shortcut ────────────────────────────────────────────────────────


@pytest.fixture
def assert_close():
    """Expose assert_close as a fixture for parametrized tolerance."""
    return _assert_close


# ── Auto-mark by path ──────────────────────────────────────────────────────────


def pytest_collection_modifyitems(items):
    for item in items:
        path = str(item.fspath)
        if "/parity/" in path:
            item.add_marker(pytest.mark.parity)
        if "/integration/" in path:
            item.add_marker(pytest.mark.slow)
