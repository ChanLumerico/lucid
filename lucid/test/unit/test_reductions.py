"""Unit tests for reduction ops: sum, mean, var, std, max, min, argmax, argmin."""

import pytest
import numpy as np
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor

_DATA = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


def _t():
    return lucid.tensor(_DATA)


class TestSum:
    def test_sum_all(self):
        assert abs(float(lucid.sum(_t()).item()) - 21.0) < 1e-4

    def test_sum_dim0(self):
        r = lucid.sum(_t(), dim=0)
        assert r.shape == (3,)
        assert_close(r, lucid.tensor([5.0, 7.0, 9.0]))

    def test_sum_dim1(self):
        r = lucid.sum(_t(), dim=1)
        assert r.shape == (2,)
        assert_close(r, lucid.tensor([6.0, 15.0]))

    def test_sum_keepdim(self):
        r = lucid.sum(_t(), dim=0, keepdim=True)
        assert r.shape == (1, 3)

    def test_sum_list_dims(self):
        t = make_tensor((2, 3, 4))
        r = lucid.sum(t, dim=[0, 2])
        assert r.shape == (3,)

    def test_sum_method(self):
        t = _t()
        assert abs(float(t.sum().item()) - 21.0) < 1e-4

    def test_sum_method_dim(self):
        t = _t()
        assert_close(t.sum(dim=0), lucid.tensor([5.0, 7.0, 9.0]))


class TestMean:
    def test_mean_all(self):
        assert abs(float(lucid.mean(_t()).item()) - 3.5) < 1e-4

    def test_mean_dim0(self):
        r = lucid.mean(_t(), dim=0)
        assert_close(r, lucid.tensor([2.5, 3.5, 4.5]))

    def test_mean_keepdim(self):
        r = lucid.mean(_t(), dim=1, keepdim=True)
        assert r.shape == (2, 1)

    def test_mean_method(self):
        t = _t()
        assert abs(float(t.mean().item()) - 3.5) < 1e-4


class TestVarStd:
    def test_var_correction1(self):
        # PyTorch default: correction=1 (Bessel)
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        var = lucid.var(t)
        np_var = np.var([1, 2, 3, 4, 5], ddof=1)
        assert abs(float(var.item()) - float(np_var)) < 1e-4

    def test_var_correction0(self):
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        var = lucid.var(t, correction=0)
        np_var = np.var([1, 2, 3, 4, 5], ddof=0)
        assert abs(float(var.item()) - float(np_var)) < 1e-4

    def test_std_correction1(self):
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        std = lucid.std(t)
        np_std = np.std([1, 2, 3, 4, 5], ddof=1)
        assert abs(float(std.item()) - float(np_std)) < 1e-4

    def test_var_dim(self):
        t = _t()
        r = lucid.var(t, dim=1)
        assert r.shape == (2,)

    def test_std_dim(self):
        t = _t()
        r = lucid.std(t, dim=0)
        assert r.shape == (3,)

    def test_var_method_matches_free_fn(self):
        t = make_tensor((4, 5))
        assert_close(t.var(dim=1), lucid.var(t, dim=1))


class TestMinMax:
    def test_max_all(self):
        assert abs(float(lucid.max(_t()).item()) - 6.0) < 1e-4

    def test_min_all(self):
        assert abs(float(lucid.min(_t()).item()) - 1.0) < 1e-4

    def test_max_dim(self):
        r = lucid.max(_t(), dim=0)
        assert_close(r, lucid.tensor([4.0, 5.0, 6.0]))

    def test_min_keepdim(self):
        r = lucid.min(_t(), dim=1, keepdim=True)
        assert r.shape == (2, 1)


class TestArgmax:
    def test_argmax_all(self):
        t = lucid.tensor([1.0, 5.0, 3.0, 2.0])
        idx = lucid.argmax(t)
        assert int(idx.item()) == 1

    def test_argmax_dim(self):
        r = lucid.argmax(_t(), dim=1)
        assert r.shape == (2,)
        assert int(r[0].item()) == 2  # row 0: max is index 2 (value 3)
        assert int(r[1].item()) == 2  # row 1: max is index 2 (value 6)

    def test_argmin_dim(self):
        r = lucid.argmin(_t(), dim=1)
        assert r.shape == (2,)
        assert int(r[0].item()) == 0  # row 0: min is index 0 (value 1)


class TestScanOps:
    def test_cumsum(self):
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        r = lucid.cumsum(t, axis=0)
        assert_close(r, lucid.tensor([1.0, 3.0, 6.0, 10.0]))

    def test_cumprod(self):
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        r = lucid.cumprod(t, axis=0)
        assert_close(r, lucid.tensor([1.0, 2.0, 6.0, 24.0]))

    def test_trace_identity(self):
        t = lucid.eye(3)
        assert abs(float(lucid.trace(t).item()) - 3.0) < 1e-4
