"""Norm / pool / dropout / loss modules."""

import math

import numpy as np
import pytest

import lucid
import lucid.nn as nn

# ── normalization ───────────────────────────────────────────────────────


class TestBatchNorm2d:
    def test_shape(self) -> None:
        m = nn.BatchNorm2d(num_features=4)
        out = m(lucid.zeros(2, 4, 8, 8))
        assert out.shape == (2, 4, 8, 8)


class TestLayerNorm:
    def test_shape(self) -> None:
        m = nn.LayerNorm([8])
        out = m(lucid.zeros(2, 8))
        assert out.shape == (2, 8)


class TestGroupNorm:
    def test_shape(self) -> None:
        m = nn.GroupNorm(num_groups=2, num_channels=4)
        out = m(lucid.zeros(2, 4, 8, 8))
        assert out.shape == (2, 4, 8, 8)


class TestRMSNorm:
    def test_shape(self) -> None:
        m = nn.RMSNorm(8)
        out = m(lucid.zeros(2, 8))
        assert out.shape == (2, 8)


class TestLocalResponseNorm:
    def test_shape(self) -> None:
        m = nn.LocalResponseNorm(size=3)
        out = m(lucid.zeros(1, 4, 4, 4))
        assert out.shape == (1, 4, 4, 4)


# ── pooling ─────────────────────────────────────────────────────────────


class TestMaxPool2d:
    def test_shape(self) -> None:
        m = nn.MaxPool2d(kernel_size=2)
        out = m(lucid.arange(0.0, 16.0, 1.0).reshape(1, 1, 4, 4))
        assert out.shape == (1, 1, 2, 2)


class TestAvgPool2d:
    def test_shape(self) -> None:
        m = nn.AvgPool2d(kernel_size=2)
        out = m(lucid.zeros(1, 1, 4, 4))
        assert out.shape == (1, 1, 2, 2)


class TestAdaptiveAvgPool:
    def test_2d(self) -> None:
        m = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        out = m(lucid.zeros(1, 3, 8, 8))
        assert out.shape == (1, 3, 2, 2)


class TestLPPool:
    def test_lppool1d(self) -> None:
        m = nn.LPPool1d(norm_type=2.0, kernel_size=2)
        out = m(lucid.arange(0.0, 8.0, 1.0).reshape(1, 1, 8))
        assert out.shape == (1, 1, 4)

    def test_lppool3d(self) -> None:
        m = nn.LPPool3d(norm_type=2.0, kernel_size=2)
        out = m(lucid.arange(0.0, 64.0, 1.0).reshape(1, 1, 4, 4, 4))
        assert out.shape == (1, 1, 2, 2, 2)


class TestMaxUnpool:
    def test_2d(self) -> None:
        m = nn.MaxUnpool2d(kernel_size=2)
        v = lucid.tensor([[[[5.0]]]])
        idx = lucid.tensor([[[[3]]]], dtype=lucid.int64)
        out = m(v, idx, output_size=(2, 2)).numpy()
        np.testing.assert_array_equal(out, [[[[0.0, 0.0], [0.0, 5.0]]]])


# ── dropout ─────────────────────────────────────────────────────────────


class TestDropout:
    def test_eval_mode_passthrough(self) -> None:
        m = nn.Dropout(p=0.5)
        m.eval()
        x = lucid.tensor([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(m(x).numpy(), [1.0, 2.0, 3.0])


class TestDropout1d:
    def test_shape(self) -> None:
        m = nn.Dropout1d(p=0.5)
        x = lucid.ones(2, 4, 8)
        assert m(x).shape == (2, 4, 8)


# ── losses ──────────────────────────────────────────────────────────────


class TestMSELoss:
    def test_zero(self) -> None:
        m = nn.MSELoss()
        x = lucid.tensor([1.0, 2.0])
        assert abs(m(x, x).item()) < 1e-6

    def test_known(self) -> None:
        m = nn.MSELoss()
        x = lucid.tensor([1.0, 2.0])
        y = lucid.tensor([2.0, 4.0])
        # mean((1-2)² + (2-4)²) = (1 + 4) / 2 = 2.5.
        assert abs(m(x, y).item() - 2.5) < 1e-6


class TestL1Loss:
    def test_known(self) -> None:
        m = nn.L1Loss()
        x = lucid.tensor([1.0, 2.0])
        y = lucid.tensor([2.0, 4.0])
        # mean(|1-2| + |2-4|) = (1+2)/2 = 1.5.
        assert abs(m(x, y).item() - 1.5) < 1e-6


class TestCrossEntropyLoss:
    def test_uniform_logits(self) -> None:
        m = nn.CrossEntropyLoss()
        # 3 classes, uniform logits, target=0 → loss = log 3.
        x = lucid.tensor([[1.0, 1.0, 1.0]])
        y = lucid.tensor([0], dtype=lucid.int64)
        assert abs(m(x, y).item() - math.log(3.0)) < 1e-5


class TestBCELoss:
    def test_known(self) -> None:
        m = nn.BCELoss()
        # input=0.5, target=1 → −log(0.5) = log 2.
        x = lucid.tensor([0.5])
        y = lucid.tensor([1.0])
        assert abs(m(x, y).item() - math.log(2.0)) < 1e-5


class TestSoftMarginAndAlias:
    def test_soft_margin_zero(self) -> None:
        m = nn.SoftMarginLoss()
        # log(1+exp(0))=log 2 when target·input=0.
        out = m(lucid.tensor([0.0]), lucid.tensor([1.0])).item()
        assert abs(out - math.log(2.0)) < 1e-6

    def test_multilabel_alias(self) -> None:
        # CamelCase alias is a subclass.
        assert issubclass(nn.MultiLabelMarginLoss, nn.MultilabelMarginLoss)


class TestTripletMarginWithDistance:
    def test_zero_when_dpos_zero(self) -> None:
        m = nn.TripletMarginWithDistanceLoss(margin=1.0)
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[1.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        assert abs(m(a, p, n).item()) < 1e-6
