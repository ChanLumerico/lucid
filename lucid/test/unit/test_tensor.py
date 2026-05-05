"""
Unit tests for Tensor construction, properties, and basic operations.
"""

import pytest
import numpy as np
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor

# ── Construction ───────────────────────────────────────────────────────────────


class TestTensorConstruction:
    def test_from_list(self):
        t = lucid.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.dtype == lucid.float32

    def test_from_nested_list(self):
        t = lucid.tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)

    def test_from_numpy_f32(self):
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        t = lucid.tensor(arr)
        assert t.shape == (1, 2)
        assert t.dtype == lucid.float32

    def test_from_numpy_f64(self):
        arr = np.ones((3, 3), dtype=np.float64)
        t = lucid.tensor(arr, dtype=lucid.float64)
        assert t.dtype == lucid.float64

    def test_requires_grad_default_false(self):
        t = lucid.tensor([1.0])
        assert t.requires_grad is False

    def test_requires_grad_true(self):
        t = lucid.tensor([1.0], requires_grad=True)
        assert t.requires_grad is True
        assert t.is_leaf is True

    def test_scalar_construction(self):
        t = lucid.tensor(3.14)
        assert t.ndim <= 1
        assert t.numel() == 1

    def test_dtype_override(self):
        t = lucid.tensor([1, 2, 3], dtype=lucid.float64)
        assert t.dtype == lucid.float64

    def test_device_cpu(self):
        t = lucid.tensor([1.0], device="cpu")
        assert t.device.type == "cpu"

    def test_copy_preserves_data(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = lucid.tensor(arr)
        arr[0] = 99.0  # mutate source — tensor should be independent
        assert float(t[0].item()) == 1.0


# ── Shape and properties ───────────────────────────────────────────────────────


class TestTensorProperties:
    def test_shape(self):
        t = make_tensor((2, 3, 4))
        assert t.shape == (2, 3, 4)

    def test_ndim(self):
        t = make_tensor((4, 5))
        assert t.ndim == 2

    def test_numel(self):
        t = make_tensor((3, 4))
        assert t.numel() == 12

    def test_dim_method(self):
        t = make_tensor((2, 3))
        assert t.dim() == 2

    def test_size_method_no_arg(self):
        t = make_tensor((2, 3, 4))
        assert t.size() == (2, 3, 4)

    def test_size_method_with_dim(self):
        t = make_tensor((2, 3, 4))
        assert t.size(1) == 3

    def test_nbytes(self):
        t = lucid.zeros(4, 4, dtype=lucid.float32)
        assert t.nbytes == 4 * 4 * 4  # 16 floats × 4 bytes

    def test_element_size_f32(self):
        t = lucid.zeros(2, dtype=lucid.float32)
        assert t.element_size() == 4

    def test_element_size_f64(self):
        t = lucid.zeros(2, dtype=lucid.float64)
        assert t.element_size() == 8

    def test_is_floating_point_f32(self):
        t = lucid.zeros(2, dtype=lucid.float32)
        assert t.is_floating_point()

    def test_is_floating_point_int(self):
        t = lucid.zeros(2, dtype=lucid.int32)
        assert not t.is_floating_point()


# ── item() ─────────────────────────────────────────────────────────────────────


class TestTensorItem:
    def test_item_scalar(self):
        t = lucid.tensor(3.14)
        assert abs(t.item() - 3.14) < 1e-5

    def test_item_1elem(self):
        t = lucid.tensor([42.0])
        assert t.item() == pytest.approx(42.0)

    def test_item_fails_multi(self):
        t = make_tensor((2,))
        with pytest.raises(Exception):
            t.item()


# ── numpy() ────────────────────────────────────────────────────────────────────


class TestTensorNumpy:
    def test_numpy_roundtrip_f32(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = lucid.tensor(arr)
        out = t.numpy()
        np.testing.assert_array_equal(out, arr)

    def test_numpy_shape_preserved(self):
        arr = np.ones((3, 4), dtype=np.float32)
        t = lucid.tensor(arr)
        assert t.numpy().shape == (3, 4)


# ── Indexing ───────────────────────────────────────────────────────────────────


class TestTensorIndexing:
    def test_integer_index(self):
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        row = t[0]
        assert row.shape == (2,)
        assert float(row[0].item()) == 1.0

    def test_slice(self):
        t = make_tensor((6,))
        s = t[2:4]
        assert s.shape == (2,)

    def test_2d_index(self):
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        val = t[1, 1]
        assert float(val.item()) == 4.0

    def test_ellipsis(self):
        t = make_tensor((2, 3, 4))
        s = t[..., 0]
        assert s.shape == (2, 3)


# ── grad properties ────────────────────────────────────────────────────────────


class TestTensorGrad:
    def test_grad_initially_none(self):
        t = lucid.tensor([1.0], requires_grad=True)
        assert t.grad is None

    def test_is_leaf(self):
        t = lucid.tensor([1.0], requires_grad=True)
        assert t.is_leaf

    def test_detach_no_grad(self):
        t = lucid.tensor([1.0, 2.0], requires_grad=True)
        d = t.detach()
        assert not d.requires_grad

    def test_clone_preserves_shape(self):
        t = make_tensor((3, 4))
        c = t.clone()
        assert c.shape == t.shape


# ── Arithmetic operators ────────────────────────────────────────────────────────


class TestTensorOps:
    def test_add_scalar(self):
        t = lucid.tensor([1.0, 2.0, 3.0])
        r = t + 1.0
        assert_close(r, lucid.tensor([2.0, 3.0, 4.0]))

    def test_sub_tensor(self):
        a = lucid.tensor([4.0, 5.0])
        b = lucid.tensor([1.0, 2.0])
        assert_close(a - b, lucid.tensor([3.0, 3.0]))

    def test_mul_scalar(self):
        t = lucid.tensor([2.0, 3.0])
        assert_close(t * 2.0, lucid.tensor([4.0, 6.0]))

    def test_neg(self):
        t = lucid.tensor([1.0, -2.0])
        assert_close(-t, lucid.tensor([-1.0, 2.0]))

    def test_matmul(self):
        a = lucid.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = lucid.tensor([[2.0], [3.0]])
        r = a @ b
        assert r.shape == (2, 1)
        assert_close(r, lucid.tensor([[2.0], [3.0]]))
