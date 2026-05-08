"""Tensor class methods + properties — non-arithmetic surface.

Arithmetic dunders / op kernels live under ``unit/ops/``; this file
covers the metadata, conversion, indexing, and property surface that
isn't tied to a specific compute kernel.
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import skip_if_unsupported
from lucid.test._helpers.compare import assert_close, assert_equal_int


class TestProperties:
    def test_shape(self, device: str) -> None:
        t = lucid.zeros(2, 3, 4, device=device)
        assert t.shape == (2, 3, 4)

    def test_ndim(self, device: str) -> None:
        assert lucid.zeros(2, 3, 4, device=device).ndim == 3
        assert lucid.zeros(device=device).ndim == 0

    def test_numel(self, device: str) -> None:
        assert lucid.zeros(2, 3, 4, device=device).numel() == 24

    def test_dtype(self, device: str) -> None:
        t = lucid.tensor([1, 2, 3], dtype=lucid.int64, device=device)
        assert t.dtype == lucid.int64

    def test_device_property(self, device: str) -> None:
        t = lucid.zeros(3, device=device)
        assert device in str(t.device).lower()

    def test_is_metal(self, device: str) -> None:
        t = lucid.zeros(3, device=device)
        assert t.is_metal == (device == "metal")

    def test_is_contiguous_default(self, device: str) -> None:
        assert lucid.zeros(3, 4, device=device).is_contiguous()


class TestNumpyConversion:
    def test_basic(self) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        arr = t.numpy()
        assert arr.shape == (2, 2)
        np.testing.assert_array_equal(arr, [[1.0, 2.0], [3.0, 4.0]])

    def test_metal_round_trip(self) -> None:
        from lucid.test._fixtures.devices import metal_available
        if not metal_available():
            pytest.skip("Metal not available")
        t = lucid.tensor([1.0, 2.0, 3.0], device="metal")
        arr = t.numpy()  # implicit GPU→CPU bridge
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


class TestTolist:
    def test_1d(self, device: str) -> None:
        assert lucid.tensor([1.0, 2.0, 3.0], device=device).tolist() == [1.0, 2.0, 3.0]

    def test_nested(self, device: str) -> None:
        assert (
            lucid.tensor([[1, 2], [3, 4]], dtype=lucid.int64, device=device).tolist()
            == [[1, 2], [3, 4]]
        )


class TestItem:
    def test_scalar(self, device: str) -> None:
        assert lucid.tensor(3.5, device=device).item() == 3.5

    def test_one_element_array(self, device: str) -> None:
        assert lucid.tensor([7.0], device=device).item() == 7.0


class TestToDevice:
    def test_cpu_to_cpu_noop(self) -> None:
        t = lucid.zeros(3)
        out = t.to(device="cpu")
        assert out.device == t.device

    def test_cpu_to_metal(self) -> None:
        from lucid.test._fixtures.devices import metal_available
        if not metal_available():
            pytest.skip("Metal not available")
        t = lucid.tensor([1.0, 2.0])
        out = t.to(device="metal")
        assert out.is_metal
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0])


class TestToDtype:
    def test_float32_to_float64(self) -> None:
        t = lucid.tensor([1.0, 2.0])
        out = t.to(dtype=lucid.float64)
        assert out.dtype == lucid.float64

    def test_float_to_int(self) -> None:
        t = lucid.tensor([1.5, 2.7, -3.1])
        out = t.to(dtype=lucid.int32)
        # Truncation toward zero is the standard cast contract.
        assert_equal_int(out, np.array([1, 2, -3], dtype=np.int32))


class TestContiguous:
    def test_already_contiguous(self, device: str) -> None:
        t = lucid.zeros(3, 4, device=device)
        out = t.contiguous()
        assert out.is_contiguous()


class TestClone:
    def test_value_equal_independent(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        c = t.clone()
        assert_close(c, t)
        # Mutate the original via reassignment — the clone should be
        # unaffected. (Actual storage independence is tested in autograd.)


class TestIndexing:
    def test_int_index(self, device: str) -> None:
        t = lucid.tensor([10.0, 20.0, 30.0], device=device)
        assert t[1].item() == 20.0

    def test_slice(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        assert_close(t[1:3], np.array([2.0, 3.0]))

    def test_2d_indexing(self, device: str) -> None:
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        assert t[0, 1].item() == 2.0
        assert_close(t[1, :], np.array([3.0, 4.0]))

    def test_negative_index(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        assert t[-1].item() == 3.0

    def test_ellipsis(self, device: str) -> None:
        t = lucid.tensor([[[1.0]]], device=device)
        assert_close(t[..., 0], np.array([[1.0]]))


class TestDunders:
    def test_len(self, device: str) -> None:
        assert len(lucid.zeros(5, 3, device=device)) == 5

    def test_repr_exists(self, device: str) -> None:
        # We don't pin the exact format — just that ``repr`` doesn't crash.
        s = repr(lucid.tensor([1.0, 2.0], device=device))
        assert isinstance(s, str)

    def test_iter(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0], device=device)
        rows = list(t)
        assert len(rows) == 3
