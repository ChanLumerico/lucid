"""Tests for pooling layers: MaxPool, AvgPool, AdaptivePool."""

import math
import pytest
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test.helpers.numerics import make_tensor


def _pool_out(i, k, s, p):
    return math.floor((i + 2 * p - k) / s) + 1


class TestMaxPool:
    def test_max_pool1d_shape(self):
        x = make_tensor((2, 4, 16))
        out = F.max_pool1d(x, kernel_size=2, stride=2)
        assert out.shape == (2, 4, 8)

    def test_max_pool2d_shape(self):
        x = make_tensor((2, 4, 8, 8))
        out = F.max_pool2d(x, kernel_size=2, stride=2)
        assert out.shape == (2, 4, 4, 4)

    def test_max_pool2d_with_padding(self):
        x = make_tensor((2, 4, 8, 8))
        out = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        assert out.shape == (2, 4, 8, 8)

    def test_nn_max_pool2d(self):
        layer = nn.MaxPool2d(kernel_size=2, stride=2)
        x = make_tensor((2, 3, 8, 8))
        assert layer(x).shape == (2, 3, 4, 4)

    def test_max_pool_is_max(self):
        x = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1,1,2,2)
        out = F.max_pool2d(x, kernel_size=2)
        assert float(out[0, 0, 0, 0].item()) == 4.0


class TestAvgPool:
    def test_avg_pool1d_shape(self):
        x = make_tensor((2, 4, 16))
        out = F.avg_pool1d(x, kernel_size=2, stride=2)
        assert out.shape == (2, 4, 8)

    def test_avg_pool2d_shape(self):
        x = make_tensor((2, 4, 8, 8))
        out = F.avg_pool2d(x, kernel_size=2, stride=2)
        assert out.shape == (2, 4, 4, 4)

    def test_avg_pool_is_average(self):
        x = lucid.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])  # (1,1,2,2)
        out = F.avg_pool2d(x, kernel_size=2)
        assert abs(float(out[0, 0, 0, 0].item()) - 4.0) < 1e-4

    def test_nn_avg_pool2d(self):
        layer = nn.AvgPool2d(kernel_size=2, stride=2)
        x = make_tensor((2, 3, 8, 8))
        assert layer(x).shape == (2, 3, 4, 4)


class TestAdaptivePool:
    def test_adaptive_avg_1d(self):
        x = make_tensor((2, 4, 16))
        out = F.adaptive_avg_pool1d(x, 4)
        assert out.shape == (2, 4, 4)

    def test_adaptive_avg_2d(self):
        x = make_tensor((2, 4, 8, 8))
        out = F.adaptive_avg_pool2d(x, (2, 2))
        assert out.shape == (2, 4, 2, 2)

    def test_adaptive_max_2d(self):
        x = make_tensor((2, 4, 8, 8))
        out = F.adaptive_max_pool2d(x, (2, 2))
        assert out.shape == (2, 4, 2, 2)

    def test_nn_adaptive_avg_2d(self):
        layer = nn.AdaptiveAvgPool2d((1, 1))
        x = make_tensor((2, 4, 8, 8))
        out = layer(x)
        assert out.shape == (2, 4, 1, 1)


class TestPoolGuards:
    """Coverage for the new Pool contract guards and adaptive non-divisible."""

    @pytest.mark.parametrize(
        "cls",
        [
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
        ],
    )
    def test_return_indices_rejected(self, cls):
        with pytest.raises(NotImplementedError, match="return_indices"):
            cls(2, return_indices=True)

    def test_adaptive_avg_2d_non_divisible(self):
        # 7×7 → 3×3 was previously rejected by the engine.
        x = make_tensor((2, 4, 7, 7))
        out = nn.AdaptiveAvgPool2d(3)(x)
        assert out.shape == (2, 4, 3, 3)

    def test_adaptive_avg_1d_non_divisible(self):
        x = make_tensor((2, 4, 11))
        out = nn.AdaptiveAvgPool1d(3)(x)
        assert out.shape == (2, 4, 3)

    def test_adaptive_avg_3d_non_divisible(self):
        x = make_tensor((1, 2, 7, 7, 5))
        out = nn.AdaptiveAvgPool3d((3, 3, 2))(x)
        assert out.shape == (1, 2, 3, 3, 2)
