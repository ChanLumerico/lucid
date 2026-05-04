"""Unit tests for unary ops: neg, abs, exp, log, trig, activations."""

import math
import pytest
import numpy as np
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor

_SHAPE = (3, 4)


def _pos(seed=0):
    """Positive-valued tensor (for log, sqrt, etc.)."""
    return make_tensor(_SHAPE, low=0.1, high=2.0, seed=seed)


def _neg_to_pos(seed=0):
    """Values near zero for activations."""
    return make_tensor(_SHAPE, low=-2.0, high=2.0, seed=seed)


class TestArith:
    def test_neg(self):
        t = make_tensor(_SHAPE)
        assert_close(lucid.neg(t), -t)

    def test_abs_positive(self):
        t = _pos()
        assert_close(lucid.abs(t), t)

    def test_abs_negative(self):
        t = make_tensor(_SHAPE, low=-2.0, high=-0.1)
        assert_close(lucid.abs(t), -t)

    def test_sign_positive(self):
        t = _pos()
        ones = lucid.ones(*_SHAPE)
        assert_close(lucid.sign(t), ones)

    def test_reciprocal(self):
        t = _pos()
        expected = lucid.ones(*_SHAPE) / t
        assert_close(lucid.reciprocal(t), expected)

    def test_square(self):
        t = make_tensor(_SHAPE, low=0.0, high=2.0)
        assert_close(lucid.square(t), t * t)


class TestExponential:
    def test_exp_at_zero(self):
        t = lucid.zeros(4)
        assert_close(lucid.exp(t), lucid.ones(4))

    def test_exp_positive(self):
        t = _pos()
        # exp is always > 0
        assert bool(lucid.min(lucid.exp(t)).item() > 0)

    def test_log_exp_inverse(self):
        t = _pos()
        assert_close(lucid.log(lucid.exp(t)), t, atol=1e-5)

    def test_log2_at_1(self):
        t = lucid.ones(4)
        assert_close(lucid.log2(t), lucid.zeros(4))

    def test_sqrt_squared(self):
        t = _pos()
        assert_close(lucid.sqrt(t * t), t, atol=1e-5)

    def test_sqrt_shape(self):
        t = _pos()
        assert lucid.sqrt(t).shape == t.shape


class TestTrig:
    def test_sin_at_zero(self):
        t = lucid.zeros(4)
        assert_close(lucid.sin(t), lucid.zeros(4))

    def test_cos_at_zero(self):
        t = lucid.zeros(4)
        assert_close(lucid.cos(t), lucid.ones(4))

    def test_sin_cos_identity(self):
        # sin²+cos²=1
        t = _neg_to_pos()
        s = lucid.sin(t)
        c = lucid.cos(t)
        ones = lucid.ones(*_SHAPE)
        assert_close(s * s + c * c, ones, atol=1e-5)

    def test_arcsin_sin_inverse(self):
        t = make_tensor(_SHAPE, low=-0.9, high=0.9)
        assert_close(lucid.arcsin(lucid.sin(t)), t, atol=1e-5)

    def test_arctan_shape(self):
        t = _neg_to_pos()
        assert lucid.arctan(t).shape == t.shape


class TestHyperbolic:
    def test_tanh_range(self):
        t = _neg_to_pos()
        out = lucid.tanh(t)
        arr = out.numpy()
        assert (arr > -1).all() and (arr < 1).all()

    def test_sinh_cosh_identity(self):
        # cosh²-sinh²=1
        t = make_tensor(_SHAPE, low=-1.0, high=1.0)
        c = lucid.cosh(t)
        s = lucid.sinh(t)
        ones = lucid.ones(*_SHAPE)
        assert_close(c * c - s * s, ones, atol=1e-4)


class TestActivations:
    def test_relu_zeros_negatives(self):
        t = make_tensor(_SHAPE, low=-2.0, high=-0.01)
        assert_close(lucid.relu(t), lucid.zeros(*_SHAPE))

    def test_relu_preserves_positives(self):
        t = _pos()
        assert_close(lucid.relu(t), t)

    def test_sigmoid_range(self):
        t = _neg_to_pos()
        out = lucid.sigmoid(t)
        arr = out.numpy()
        assert (arr > 0).all() and (arr < 1).all()

    def test_sigmoid_at_zero_is_half(self):
        t = lucid.zeros(4)
        half = lucid.full((4,), 0.5)
        assert_close(lucid.sigmoid(t), half, atol=1e-6)

    def test_silu_shape(self):
        t = _neg_to_pos()
        assert lucid.silu(t).shape == t.shape

    def test_gelu_shape(self):
        t = _neg_to_pos()
        assert lucid.gelu(t).shape == t.shape

    def test_softplus_positive(self):
        t = _neg_to_pos()
        out = lucid.softplus(t)
        assert bool(lucid.min(out).item() > 0)


class TestRounding:
    def test_floor(self):
        t = lucid.tensor([1.7, -1.3, 2.0])
        expected = lucid.tensor([1.0, -2.0, 2.0])
        assert_close(lucid.floor(t), expected)

    def test_ceil(self):
        t = lucid.tensor([1.1, -1.9, 2.0])
        expected = lucid.tensor([2.0, -1.0, 2.0])
        assert_close(lucid.ceil(t), expected)

    def test_round(self):
        t = lucid.tensor([1.4, 1.6, -1.5])
        out = lucid.round(t)
        assert out.shape == t.shape
