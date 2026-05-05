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
        expected = 0.5 * 0.5**2  # quadratic for |error| < delta
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

    @pytest.mark.parametrize(
        "fn",
        [
            F.mse_loss,
            F.l1_loss,
            F.huber_loss,
            F.smooth_l1_loss,
            F.binary_cross_entropy_with_logits,
        ],
    )
    def test_invalid_reduction_rejected(self, fn):
        x = make_tensor((4,))
        t = make_tensor((4,), seed=1)
        with pytest.raises(ValueError, match="reduction"):
            fn(x, t, reduction="bogus")


class TestCrossEntropyContract:
    def test_ignore_index_excludes_samples(self):
        # If we ignore samples whose target is ignore_index, the resulting
        # mean loss should equal the mean over the surviving samples.
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6, 5)).astype(np.float32)
        target = np.array([0, 1, -1, 2, -1, 3], dtype=np.int32)
        full = F.cross_entropy(
            lucid.tensor(x),
            lucid.tensor(target, dtype=lucid.int32),
            ignore_index=-1,
            reduction="mean",
        ).item()
        # Manual ref: take only valid rows, average their per-sample CE.
        valid_rows = [i for i, t in enumerate(target) if t != -1]
        per_sample = F.cross_entropy(
            lucid.tensor(x),
            lucid.tensor(
                np.where(target == -1, 0, target).astype(np.int32), dtype=lucid.int32
            ),
            reduction="none",
        ).numpy()
        manual = per_sample[valid_rows].mean()
        np.testing.assert_allclose(full, manual, atol=1e-5)

    def test_label_smoothing_increases_loss(self):
        # On a confident prediction, label smoothing penalises certainty,
        # so the loss should rise.
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 5)).astype(np.float32) * 5.0
        target = np.array([0, 1, 2, 3], dtype=np.int32)
        base = F.cross_entropy(
            lucid.tensor(x),
            lucid.tensor(target, dtype=lucid.int32),
            label_smoothing=0.0,
        ).item()
        smoothed = F.cross_entropy(
            lucid.tensor(x),
            lucid.tensor(target, dtype=lucid.int32),
            label_smoothing=0.1,
        ).item()
        assert smoothed > base

    def test_label_smoothing_validation(self):
        x = make_tensor((4, 5))
        t = lucid.tensor([0, 1, 2, 3], dtype=lucid.int32)
        with pytest.raises(ValueError, match="label_smoothing"):
            F.cross_entropy(x, t, label_smoothing=1.0)
        with pytest.raises(ValueError, match="label_smoothing"):
            F.cross_entropy(x, t, label_smoothing=-0.1)

    def test_weight_changes_loss(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 5)).astype(np.float32)
        target = np.array([0, 1, 2, 3], dtype=np.int32)
        base = F.cross_entropy(
            lucid.tensor(x),
            lucid.tensor(target, dtype=lucid.int32),
        ).item()
        # Heavy weight on class 0 should change the value.
        weighted = F.cross_entropy(
            lucid.tensor(x),
            lucid.tensor(target, dtype=lucid.int32),
            weight=lucid.tensor(np.array([10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)),
        ).item()
        assert abs(weighted - base) > 1e-3


class TestNLLLossContract:
    def test_ignore_index(self):
        rng = np.random.default_rng(0)
        log_p = rng.standard_normal((4, 5)).astype(np.float32)
        log_p = log_p - np.log(np.exp(log_p).sum(axis=1, keepdims=True))
        target = np.array([0, -1, 2, -1], dtype=np.int32)
        l = F.nll_loss(
            lucid.tensor(log_p),
            lucid.tensor(target, dtype=lucid.int32),
            ignore_index=-1,
            reduction="mean",
        ).item()
        # Manual: average only valid rows.
        per = F.nll_loss(
            lucid.tensor(log_p),
            lucid.tensor(
                np.where(target == -1, 0, target).astype(np.int32), dtype=lucid.int32
            ),
            reduction="none",
        ).numpy()
        manual = per[[0, 2]].mean()
        np.testing.assert_allclose(l, manual, atol=1e-5)

    def test_module_signature(self):
        # NLLLoss now accepts weight + ignore_index.
        loss_fn = nn.NLLLoss(
            weight=lucid.tensor(np.array([0.5, 1.0, 2.0, 1.5, 0.8], dtype=np.float32)),
            ignore_index=2,
        )
        log_p = make_tensor((4, 5))
        target = lucid.tensor([0, 1, 2, 3], dtype=lucid.int32)
        out = loss_fn(log_p, target)
        assert out.shape == ()
        assert "ignore_index=2" in repr(loss_fn)


class TestBCEContract:
    def test_module_accepts_weight(self):
        loss_fn = nn.BCELoss(weight=lucid.ones((4,)))
        x = lucid.sigmoid(make_tensor((4,)))
        t = lucid.tensor([0.0, 1.0, 0.0, 1.0])
        assert loss_fn(x, t).shape == ()

    def test_bce_with_logits_pos_weight(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((8, 4)).astype(np.float32)
        t = rng.integers(0, 2, (8, 4)).astype(np.float32)
        pw = np.array([2.0, 1.0, 0.5, 1.5], dtype=np.float32)
        with_pw = F.binary_cross_entropy_with_logits(
            lucid.tensor(x),
            lucid.tensor(t),
            pos_weight=lucid.tensor(pw),
        ).item()
        without = F.binary_cross_entropy_with_logits(
            lucid.tensor(x),
            lucid.tensor(t),
        ).item()
        # pos_weight ≠ 1 must change the loss value.
        assert abs(with_pw - without) > 1e-4


class TestKLDivContract:
    def test_batchmean_equals_sum_over_batch_size(self):
        rng = np.random.default_rng(0)
        log_q = np.log(np.array([[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]], dtype=np.float32))
        p = np.array([[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]], dtype=np.float32)
        s = F.kl_div(lucid.tensor(log_q), lucid.tensor(p), reduction="sum").item()
        bm = F.kl_div(
            lucid.tensor(log_q), lucid.tensor(p), reduction="batchmean"
        ).item()
        np.testing.assert_allclose(bm, s / 2.0, atol=1e-5)
