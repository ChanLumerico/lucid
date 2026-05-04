"""Tests for loss functions: MSE, CE, BCE, NLL, Huber, L1."""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


class TestMSELoss:
    def test_perfect_prediction_zero_loss(self):
        pred = lucid.tensor([1.0, 2.0, 3.0])
        target = lucid.tensor([1.0, 2.0, 3.0])
        loss = F.mse_loss(pred, target)
        assert abs(float(loss.item())) < 1e-6

    def test_loss_positive(self):
        pred = make_tensor((4,))
        target = make_tensor((4,), seed=1)
        loss = F.mse_loss(pred, target)
        assert float(loss.item()) >= 0.0

    def test_scalar_output(self):
        pred = make_tensor((4, 6))
        target = make_tensor((4, 6), seed=1)
        loss = F.mse_loss(pred, target)
        assert loss.shape == ()

    def test_nn_mse_loss(self):
        loss_fn = nn.MSELoss()
        pred = make_tensor((4,))
        target = make_tensor((4,), seed=1)
        loss = loss_fn(pred, target)
        assert float(loss.item()) >= 0.0


class TestBCELoss:
    def test_output_is_scalar(self):
        pred = lucid.sigmoid(make_tensor((8,)))
        target = lucid.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        loss = F.binary_cross_entropy(pred, target)
        assert loss.shape == ()

    def test_perfect_pred_low_loss(self):
        target = lucid.tensor([1.0, 1.0, 1.0])
        pred = lucid.tensor([0.9999, 0.9999, 0.9999])
        loss = F.binary_cross_entropy(pred, target)
        assert float(loss.item()) < 0.01


class TestCrossEntropyLoss:
    def test_output_is_scalar(self):
        logits = make_tensor((4, 10))
        targets = lucid.tensor([0, 3, 7, 2], dtype=lucid.int32)
        loss = F.cross_entropy(logits, targets)
        assert loss.shape == ()

    def test_loss_non_negative(self):
        logits = make_tensor((8, 5))
        targets = lucid.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=lucid.int32)
        loss = F.cross_entropy(logits, targets)
        assert float(loss.item()) >= 0.0

    def test_nn_cross_entropy(self):
        loss_fn = nn.CrossEntropyLoss()
        logits = make_tensor((4, 10))
        targets = lucid.tensor([0, 3, 7, 2], dtype=lucid.int32)
        loss = loss_fn(logits, targets)
        assert loss.shape == ()


class TestHuberLoss:
    def test_quadratic_below_delta(self):
        pred = lucid.tensor([0.5])
        target = lucid.tensor([0.0])
        loss = F.huber_loss(pred, target, delta=1.0)
        expected = 0.5 * 0.5 ** 2  # quadratic for |error| < delta
        assert abs(float(loss.item()) - expected) < 1e-4

    def test_linear_above_delta(self):
        pred = lucid.tensor([2.0])
        target = lucid.tensor([0.0])
        loss = F.huber_loss(pred, target, delta=1.0)
        expected = 1.0 * (2.0 - 0.5 * 1.0)  # linear for |error| > delta
        assert abs(float(loss.item()) - expected) < 1e-4


class TestReductionModes:
    def test_mse_reduction_sum(self):
        pred = lucid.tensor([1.0, 2.0, 3.0])
        target = lucid.tensor([0.0, 0.0, 0.0])
        mean_loss = F.mse_loss(pred, target)
        sum_loss = F.mse_loss(pred, target, reduction="sum")
        # sum = mean * n
        assert abs(float(sum_loss.item()) - 3.0 * float(mean_loss.item())) < 1e-4

    def test_mse_reduction_none(self):
        pred = make_tensor((4,))
        target = make_tensor((4,), seed=1)
        loss = F.mse_loss(pred, target, reduction="none")
        assert loss.shape == (4,)
