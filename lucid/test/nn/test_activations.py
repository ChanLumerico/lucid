"""Unit tests for nn.functional activation functions."""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
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


class TestSoftplus:
    def test_default_matches_log1p_exp(self) -> None:
        # Default beta=1, threshold=20 — straight ``log(1 + exp(x))``.
        layer: nn.Softplus = nn.Softplus()
        x: lucid.Tensor = lucid.tensor([0.0, 1.0, 2.0, -1.0])
        np.testing.assert_allclose(
            layer(x).numpy(),
            np.log1p(np.exp(np.array([0.0, 1.0, 2.0, -1.0]))),
            atol=1e-5,
        )

    def test_beta_scales_input(self) -> None:
        # ``softplus(x; beta) = (1/beta) * log(1 + exp(beta * x))``
        layer: nn.Softplus = nn.Softplus(beta=2.0)
        x_np: np.ndarray = np.array([0.5, 1.0])
        x: lucid.Tensor = lucid.tensor(x_np.astype(np.float32))
        expected: np.ndarray = np.log1p(np.exp(2.0 * x_np)) / 2.0
        np.testing.assert_allclose(layer(x).numpy(), expected, atol=1e-5)

    def test_threshold_falls_back_to_identity(self) -> None:
        # For ``beta * x > threshold`` the function returns ``x`` directly so
        # the ``exp`` term doesn't overflow.
        layer: nn.Softplus = nn.Softplus(beta=1.0, threshold=20.0)
        x: lucid.Tensor = lucid.tensor([100.0, 0.5])
        out: np.ndarray = layer(x).numpy()
        # 100.0 hits the identity branch; 0.5 follows the analytic formula.
        assert out[0] == pytest.approx(100.0)
        assert out[1] == pytest.approx(np.log1p(np.exp(0.5)), abs=1e-5)
