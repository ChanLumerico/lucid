"""Bridge surface: ``from_numpy`` / ``.numpy()`` / DLPack."""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available


class TestFromNumpy:
    def test_dtype_inferred_from_array(self) -> None:
        for np_dt, expected in [
            (np.float32, lucid.float32),
            (np.float64, lucid.float64),
            (np.int32, lucid.int32),
            (np.int64, lucid.int64),
            (np.bool_, lucid.bool_),
        ]:
            arr = np.array([1, 0, 1], dtype=np_dt)
            t = lucid.from_numpy(arr)
            assert t.dtype == expected, f"{np_dt} → {t.dtype}, expected {expected}"

    def test_shape_preserved(self) -> None:
        arr = np.zeros((3, 4, 5), dtype=np.float32)
        t = lucid.from_numpy(arr)
        assert t.shape == (3, 4, 5)


class TestToNumpy:
    def test_basic(self) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0])
        arr = t.numpy()
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_metal_implicit_cpu_bridge(self) -> None:
        if not metal_available():
            pytest.skip("Metal not available")
        t = lucid.tensor([1.0, 2.0, 3.0], device="metal")
        arr = t.numpy()
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


class TestDLPackProtocol:
    def test_dlpack_device_is_cpu(self) -> None:
        # ``__dlpack_device__`` always reports CPU because the export
        # routes through numpy.  ``(1, 0)`` is the kDLCPU device id.
        t = lucid.tensor([1.0, 2.0])
        assert t.__dlpack_device__() == (1, 0)

    def test_dlpack_returns_capsule(self) -> None:
        t = lucid.tensor([1.0, 2.0])
        cap = t.__dlpack__()
        assert type(cap).__name__ == "PyCapsule"

    def test_to_dlpack_helper(self) -> None:
        t = lucid.tensor([1.0, 2.0])
        cap = lucid.to_dlpack(t)
        assert type(cap).__name__ == "PyCapsule"

    def test_round_trip_through_numpy(self) -> None:
        t = lucid.tensor([1.5, 2.5, 3.5])
        # numpy's ``np.from_dlpack`` reads the protocol off the
        # producer object directly.
        arr = np.from_dlpack(t)
        np.testing.assert_array_equal(arr, [1.5, 2.5, 3.5])

    def test_from_dlpack_consumes_numpy(self) -> None:
        arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        t = lucid.from_dlpack(arr)
        np.testing.assert_array_equal(t.numpy(), [10.0, 20.0, 30.0])

    def test_metal_export_via_cpu_bridge(self) -> None:
        if not metal_available():
            pytest.skip("Metal not available")
        g = lucid.tensor([1.0, 2.0], device="metal")
        arr = np.from_dlpack(g)
        np.testing.assert_array_equal(arr, [1.0, 2.0])
