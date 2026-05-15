"""``lucid.metal`` — Metal-stream specific behaviour.

Most of the real Metal validation lives in cross-device tests under
``unit/device/`` and ``unit/ops/`` — this file is the home for any
metal-only edge cases (allocation, sync, driver-level surface).
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available


@pytest.fixture(autouse=True)
def _require_metal() -> None:
    if not metal_available():
        pytest.skip("Metal not available on this host")


class TestMetalSurface:
    def test_module_present(self) -> None:
        # ``lucid.metal`` is always importable; behaviour just degrades
        # to no-op when Metal isn't there.
        assert hasattr(lucid, "metal")


class TestMetalAllocation:
    def test_zeros(self) -> None:
        t = lucid.zeros(8, device="metal")
        assert t.is_metal
        np.testing.assert_array_equal(t.numpy(), np.zeros(8))

    def test_round_trip(self) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = a.to(device="metal")
        c = b.to(device="cpu")
        np.testing.assert_array_equal(a.numpy(), c.numpy())


class TestMetalArith:
    def test_add(self) -> None:
        a = lucid.tensor([1.0, 2.0], device="metal")
        b = lucid.tensor([3.0, 4.0], device="metal")
        np.testing.assert_array_equal((a + b).numpy(), [4.0, 6.0])

    def test_matmul(self) -> None:
        a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device="metal")
        b = lucid.tensor([[5.0, 6.0], [7.0, 8.0]], device="metal")
        np.testing.assert_array_equal((a @ b).numpy(), [[19.0, 22.0], [43.0, 50.0]])
