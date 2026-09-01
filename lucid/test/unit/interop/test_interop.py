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


class TestArrayProtocol:
    """``np.asarray(t)`` has to produce numbers, not a box of Tensors.

    Without ``__array__`` NumPy falls back to the sequence protocol and
    builds an ``dtype=object`` array of 0-d Tensors.  Nothing raises —
    the result just stops being numeric, so it is the one bridge failure
    a smoke test would not notice.
    """

    def test_asarray_is_numeric(self) -> None:
        arr = np.asarray(lucid.ones(2, 3))
        assert arr.dtype == np.float32
        assert arr.shape == (2, 3)
        assert arr.sum() == 6.0

    def test_asarray_matches_numpy_method(self) -> None:
        t = lucid.linspace(0.0, 1.0, 6).reshape(2, 3)
        assert np.array_equal(np.asarray(t), t.numpy())

    def test_dtype_request_is_honoured(self) -> None:
        assert np.asarray(lucid.ones(2), dtype=np.float64).dtype == np.float64

    def test_copy_false_is_refused(self) -> None:
        # A Lucid tensor has to be read out through the host, so the
        # no-copy contract NumPy 2 defines cannot be met.
        with pytest.raises(ValueError, match="copy=False"):
            np.array(lucid.ones(2), copy=False)

    @pytest.mark.skipif(not metal_available(), reason="needs Metal")
    def test_metal_goes_through_the_cpu_bridge(self) -> None:
        arr = np.asarray(lucid.ones(2, 3).to("metal"))
        assert arr.dtype == np.float32
        assert arr.sum() == 6.0


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
