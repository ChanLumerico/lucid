"""
Tests for basic Tensor creation, properties, and repr.
"""

import pytest
import numpy as np
import lucid
from lucid._tensor.tensor import Tensor
from conftest import assert_close


class TestTensorCreation:
    def test_zeros(self):
        t = lucid.zeros(3, 4)
        assert t.shape == (3, 4)
        assert np.all(t.numpy() == 0.0)

    def test_ones(self):
        t = lucid.ones(2, 3)
        assert t.shape == (2, 3)
        assert np.all(t.numpy() == 1.0)

    def test_randn_shape(self):
        t = lucid.randn(5, 6)
        assert t.shape == (5, 6)

    def test_arange(self):
        t = lucid.arange(5)
        assert_close(t.numpy(), np.arange(5, dtype=np.float32))

    def test_from_numpy(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = lucid.from_numpy(arr)
        assert_close(t.numpy(), arr)

    def test_tensor_from_list(self):
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.shape == (2, 2)

    def test_scalar(self):
        t = lucid.tensor([3.14])
        assert t.shape == (1,)
        assert abs(t.item() - 3.14) < 1e-5


class TestTensorProperties:
    def test_shape(self):
        t = lucid.zeros(2, 3, 4)
        assert t.shape == (2, 3, 4)
        assert t.ndim == 3

    def test_dtype(self):
        t = lucid.zeros(3)
        assert t.dtype == lucid.float32

    def test_device(self):
        t = lucid.zeros(3)
        assert "cpu" in str(t.device)

    def test_numel(self):
        t = lucid.zeros(2, 3, 4)
        assert t.numel() == 24

    def test_nbytes(self):
        t = lucid.zeros(2, 3)  # float32 = 4 bytes each
        assert t.nbytes == 24

    def test_element_size(self):
        t = lucid.zeros(3)
        assert t.element_size() == 4

    def test_is_floating_point(self):
        t = lucid.zeros(3)
        assert t.is_floating_point()

    def test_T_property(self):
        t = lucid.randn(3, 4)
        assert t.T.shape == (4, 3)
        # T should be a property, not a method
        assert isinstance(Tensor.__dict__["T"], property)

    def test_mT_property(self):
        t = lucid.randn(2, 3, 4)
        assert t.mT.shape == (2, 4, 3)
        assert isinstance(Tensor.__dict__["mT"], property)


class TestTensorIter:
    def test_iter_1d(self):
        t = lucid.tensor([1.0, 2.0, 3.0])
        items = list(t)
        assert len(items) == 3
        assert all(it.shape == () for it in items)

    def test_iter_2d(self):
        t = lucid.randn(4, 5)
        rows = list(t)
        assert len(rows) == 4
        assert all(r.shape == (5,) for r in rows)

    def test_iter_0d_raises(self):
        import numpy as np
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _wrap
        # Create a true 0-d numpy array
        arr = np.array(1.0, dtype=np.float32)
        impl = _C_engine.TensorImpl(arr, _C_engine.Device.CPU, False)
        t = _wrap(impl)
        if t.ndim == 0:
            with pytest.raises(TypeError, match="0-d"):
                list(t)
        else:
            pytest.skip("Engine promotes 0-d arrays to 1-d; 0-d iter test not applicable")


class TestNewMethods:
    def test_new_zeros(self):
        x = lucid.randn(3, 4)
        z = x.new_zeros(2, 5)
        assert z.shape == (2, 5)
        assert z.dtype == x.dtype
        assert np.all(z.numpy() == 0.0)

    def test_new_ones(self):
        x = lucid.randn(3)
        o = x.new_ones(4, 4)
        assert o.shape == (4, 4)
        assert np.all(o.numpy() == 1.0)

    def test_new_full(self):
        x = lucid.randn(3)
        f = x.new_full((2, 3), 7.0)
        assert np.all(f.numpy() == 7.0)

    def test_new_tensor(self):
        x = lucid.randn(3)
        t = x.new_tensor([1.0, 2.0])
        assert t.shape == (2,)


class TestTensorRepr:
    def test_repr_shape(self):
        t = lucid.zeros(2, 3)
        r = repr(t)
        assert "tensor" in r.lower() or "Tensor" in r

    def test_format(self):
        t = lucid.ones(1)
        s = f"{t:.2f}"
        assert "1.00" in s
