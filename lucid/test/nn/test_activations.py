"""Unit tests for nn.functional activation functions."""

import pytest
import numpy as np
import lucid
import lucid.nn.functional as F
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor

_SHAPE = (4, 8)


class TestReLU:
    def test_zeros_negatives(self):
        t = make_tensor(_SHAPE, low=-2.0, high=-0.01)
        out = F.relu(t)
        assert (out.numpy() == 0).all()

    def test_keeps_positives(self):
        t = make_tensor(_SHAPE, low=0.01, high=2.0)
        assert_close(F.relu(t), t)

    def test_shape(self):
        t = make_tensor(_SHAPE)
        assert F.relu(t).shape == _SHAPE


class TestGELU:
    def test_shape(self):
        t = make_tensor(_SHAPE)
        assert F.gelu(t).shape == _SHAPE

    def test_gelu_positive_at_large_x(self):
        t = lucid.full(_SHAPE, 5.0)
        out = F.gelu(t)
        assert (out.numpy() > 4.9).all()

    def test_gelu_near_zero_at_neg_large(self):
        t = lucid.full(_SHAPE, -5.0)
        out = F.gelu(t)
        assert (np.abs(out.numpy()) < 0.01).all()

    def test_gelu_tanh_mode(self):
        t = make_tensor(_SHAPE)
        out = F.gelu(t, approximate="tanh")
        assert out.shape == _SHAPE


class TestSigmoid:
    def test_range(self):
        t = make_tensor(_SHAPE, low=-5.0, high=5.0)
        out = F.sigmoid(t).numpy()
        assert (out > 0).all() and (out < 1).all()

    def test_at_zero(self):
        t = lucid.zeros(*_SHAPE)
        half = lucid.full(_SHAPE, 0.5)
        assert_close(F.sigmoid(t), half, atol=1e-6)


class TestSoftmax:
    def test_sums_to_one(self):
        t = make_tensor((3, 5))
        out = F.softmax(t, dim=-1)
        row_sums = lucid.sum(out, dim=1).numpy()
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-5)

    def test_shape(self):
        t = make_tensor((2, 4, 6))
        assert F.softmax(t, dim=2).shape == (2, 4, 6)

    def test_log_softmax_shape(self):
        t = make_tensor((3, 5))
        out = F.log_softmax(t, dim=-1)
        assert out.shape == (3, 5)
        # log-softmax values should be <= 0
        assert (out.numpy() <= 0).all()


class TestLeakyReLU:
    def test_positive_unchanged(self):
        t = make_tensor(_SHAPE, low=0.1, high=2.0)
        assert_close(F.leaky_relu(t), t)

    def test_negative_scaled(self):
        t = lucid.full(_SHAPE, -1.0)
        out = F.leaky_relu(t, negative_slope=0.1)
        expected = lucid.full(_SHAPE, -0.1)
        assert_close(out, expected, atol=1e-5)


class TestELU:
    def test_positive_unchanged(self):
        t = make_tensor(_SHAPE, low=0.1, high=2.0)
        assert_close(F.elu(t), t)

    def test_negative_saturates(self):
        t = lucid.full(_SHAPE, -100.0)
        out = F.elu(t, alpha=1.0)
        expected = lucid.full(_SHAPE, -1.0)
        assert_close(out, expected, atol=1e-4)


class TestOtherActivations:
    @pytest.mark.parametrize("fn_name", ["silu", "mish", "selu"])
    def test_shape_preserved(self, fn_name):
        fn = getattr(F, fn_name)
        t = make_tensor(_SHAPE)
        assert fn(t).shape == _SHAPE

    def test_hardswish_shape(self):
        t = make_tensor(_SHAPE)
        assert F.hardswish(t).shape == _SHAPE

    def test_hardsigmoid_range(self):
        t = make_tensor(_SHAPE, low=-5.0, high=5.0)
        out = F.hardsigmoid(t).numpy()
        assert (out >= 0).all() and (out <= 1).all()
