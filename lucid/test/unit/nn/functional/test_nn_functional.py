"""nn.functional — numerical correctness against closed-form refs."""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F


class TestActivationsF:
    def test_relu(self) -> None:
        out = F.relu(lucid.tensor([-1.0, 0.0, 1.0])).numpy()
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0])

    def test_softmax_uniform(self) -> None:
        out = F.softmax(lucid.tensor([[1.0, 1.0, 1.0]]), dim=1).numpy()
        np.testing.assert_allclose(out, [[1 / 3] * 3], atol=1e-6)

    def test_log_softmax_sum_exp_one(self) -> None:
        out = F.log_softmax(lucid.tensor([[1.0, 1.0, 1.0]]), dim=1).numpy()
        # exp of log_softmax should sum to 1.
        assert abs(np.exp(out).sum() - 1.0) < 1e-6

    def test_gelu(self) -> None:
        # gelu(0) = 0.
        assert abs(F.gelu(lucid.tensor([0.0])).item()) < 1e-6

    def test_silu(self) -> None:
        # silu(x) = x * sigmoid(x); silu(0) = 0.
        assert abs(F.silu(lucid.tensor([0.0])).item()) < 1e-6

    def test_softplus(self) -> None:
        # softplus(0) = log 2.
        assert abs(F.softplus(lucid.tensor([0.0])).item() - math.log(2.0)) < 1e-5


class TestPooling:
    def test_max_pool2d_shape(self) -> None:
        out = F.max_pool2d(lucid.zeros(1, 1, 4, 4), kernel_size=2)
        assert out.shape == (1, 1, 2, 2)

    def test_avg_pool2d_value(self) -> None:
        # Average of [[1, 1], [1, 1]] over 2x2 → 1.0.
        x = lucid.ones(1, 1, 2, 2)
        out = F.avg_pool2d(x, kernel_size=2)
        assert out.item() == 1.0


class TestNormalization:
    def test_layer_norm_unit_variance(self) -> None:
        # After layer_norm the output dim has zero mean / unit variance.
        x = lucid.tensor([[1.0, 2.0, 3.0, 4.0]])
        gamma = lucid.ones(4)
        beta = lucid.zeros(4)
        out = F.layer_norm(x, [4], weight=gamma, bias=beta).numpy()
        assert abs(out.mean()) < 1e-5
        assert abs(out.std() - 1.0) < 1e-2  # bessel correction in some impls.


class TestLossesF:
    def test_mse(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        y = lucid.tensor([2.0, 4.0])
        assert abs(F.mse_loss(x, y).item() - 2.5) < 1e-6

    def test_cross_entropy_uniform(self) -> None:
        x = lucid.tensor([[1.0, 1.0, 1.0]])
        y = lucid.tensor([0], dtype=lucid.int64)
        assert abs(F.cross_entropy(x, y).item() - math.log(3.0)) < 1e-5


class TestGumbelSoftmax:
    def test_soft_sums_to_one(self) -> None:
        out = F.gumbel_softmax(
            lucid.tensor([[1.0, 2.0, 3.0]]), tau=1.0, hard=False
        ).numpy()
        assert abs(out.sum() - 1.0) < 1e-5

    def test_hard_one_hot(self) -> None:
        out = F.gumbel_softmax(
            lucid.tensor([[1.0, 2.0, 3.0]]), tau=1.0, hard=True
        ).numpy()
        assert out.sum() == 1.0


class TestTripletWithDistance:
    def test_zero_when_dpos_zero(self) -> None:
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[1.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        assert (
            abs(F.triplet_margin_with_distance_loss(a, p, n, margin=1.0).item()) < 1e-6
        )
