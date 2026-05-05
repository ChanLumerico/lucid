"""Unit tests for device placement and device-related errors."""

import pytest
import lucid
from lucid.test.helpers.numerics import make_tensor


class TestCpuDevice:
    def test_tensor_on_cpu(self):
        t = lucid.tensor([1.0], device="cpu")
        assert "cpu" in str(t.device).lower()

    def test_zeros_on_cpu(self):
        t = lucid.zeros(3, device="cpu")
        assert "cpu" in str(t.device).lower()

    def test_to_cpu(self):
        t = make_tensor((4,))
        t2 = t.to("cpu")
        assert "cpu" in str(t2.device).lower()

    def test_make_tensor_default_cpu(self):
        t = make_tensor((2, 3))
        assert "cpu" in str(t.device).lower()


class TestDeviceMethods:
    def test_cpu_method(self):
        t = make_tensor((4,))
        t_cpu = t.cpu()
        assert "cpu" in str(t_cpu.device).lower()

    def test_is_metal_false_for_cpu(self):
        t = make_tensor((2,))
        assert not t.is_metal


class TestDeviceErrors:
    def test_wrong_device_string_raises(self):
        with pytest.raises(Exception):
            lucid.tensor([1.0], device="cuda")  # not supported

    def test_numpy_requires_cpu(self):
        t = make_tensor((4,))
        arr = t.numpy()
        import numpy as np

        assert isinstance(arr, np.ndarray)
