"""Device fixtures — drives CPU/Metal cross-validation.

The default ``device`` fixture parametrizes every requesting test over
the available compute streams.  Metal is detected by attempting a
zero-cost allocation and skipped at parametrize time when unavailable.

Some dtypes (``float64`` / ``bfloat16``) are not supported on Metal —
``skip_if_unsupported(device, dtype)`` is the canonical guard for the
device × dtype cross-product.
"""

import functools
from collections.abc import Iterator
from typing import Any

import pytest

import lucid


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


# Dtypes MLX doesn't support on Metal.  Tests that hit this combo
# should call ``skip_if_unsupported(device, dtype)`` before doing any
# allocation so the parametrize-cross-product cleanly skips the cells
# that can't run.
_METAL_UNSUPPORTED_DTYPES: frozenset[Any] = frozenset(
    {
        lucid.float64,
        lucid.bfloat16,
        lucid.complex64,  # complex on Metal is opt-in via specific kernels.
    }
)


def skip_if_unsupported(device: str, dtype: Any) -> None:
    """Skip the calling test when ``(device, dtype)`` isn't supported.

    Currently the only constraint is "Metal can't do float64 / bfloat16
    / complex64" — call this helper at the top of any test that takes
    both ``device`` and a dtype fixture.
    """
    if device == "metal" and dtype in _METAL_UNSUPPORTED_DTYPES:
        pytest.skip(f"{dtype} not supported on Metal")
