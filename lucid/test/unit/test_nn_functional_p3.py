"""Unit tests for the P3 ``lucid.nn.functional`` additions."""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

# ── activations ────────────────────────────────────────────────────────────


class TestHardtanh:
    def test_default_range(self) -> None:
        out = F.hardtanh(lucid.tensor([-2.0, -0.5, 0.5, 2.0])).numpy()
        np.testing.assert_array_equal(out, [-1.0, -0.5, 0.5, 1.0])

    def test_custom_range(self) -> None:
        out = F.hardtanh(
            lucid.tensor([-3.0, 0.0, 3.0]), min_val=-2.0, max_val=2.0
        ).numpy()
        np.testing.assert_array_equal(out, [-2.0, 0.0, 2.0])


class TestLogsigmoid:
    def test_zero(self) -> None:
        # log σ(0) = log(0.5) = -log(2)
        assert abs(F.logsigmoid(lucid.tensor([0.0])).item() + math.log(2.0)) < 1e-6

    def test_large_positive_stable(self) -> None:
        # log σ(1000) ≈ 0; the unstable form (log of sigmoid) would
        # underflow, the softplus form must not.
        out = F.logsigmoid(lucid.tensor([1000.0])).item()
        assert abs(out) < 1e-3
        assert math.isfinite(out)


class TestSoftsign:
    def test_zero(self) -> None:
        assert F.softsign(lucid.tensor([0.0])).item() == 0.0

    def test_known_values(self) -> None:
        out = F.softsign(lucid.tensor([1.0, -2.0])).numpy()
        np.testing.assert_allclose(out, [0.5, -2.0 / 3.0], atol=1e-6)


class TestThreshold:
    def test_replaces_below(self) -> None:
        # x > threshold ? x : value  (strict > , matches reference semantics)
        out = F.threshold(
            lucid.tensor([-1.0, 0.0, 1.0, 2.0]), threshold=0.5, value=-9.0
        ).numpy()
        np.testing.assert_array_equal(out, [-9.0, -9.0, 1.0, 2.0])


# ── normalization ──────────────────────────────────────────────────────────


class TestLocalResponseNorm:
    def test_preserves_shape(self) -> None:
        x = lucid.arange(0.0, 16.0, 1.0).reshape(1, 4, 2, 2)
        for size in (1, 2, 3, 4):
            assert F.local_response_norm(x, size=size).shape == (1, 4, 2, 2)

    def test_all_zeros(self) -> None:
        # ``y = x · (k + α/n · 0)^-β = x · k^-β``.  At k=1.0 default this
        # is just ``x``, regardless of size.
        x = lucid.zeros(1, 4, 2, 2)
        out = F.local_response_norm(x, size=3)
        np.testing.assert_array_equal(out.numpy(), np.zeros((1, 4, 2, 2)))


# ── pooling ────────────────────────────────────────────────────────────────


class TestLpPool:
    def test_lp_pool1d_shape(self) -> None:
        x = lucid.arange(0.0, 8.0, 1.0).reshape(1, 1, 8)
        assert F.lp_pool1d(x, norm_type=2.0, kernel_size=2).shape == (1, 1, 4)

    def test_lp_pool2d_shape(self) -> None:
        x = lucid.arange(0.0, 16.0, 1.0).reshape(1, 1, 4, 4)
        assert F.lp_pool2d(x, norm_type=2.0, kernel_size=2).shape == (1, 1, 2, 2)

    def test_lp_pool_p1_eq_sum_abs(self) -> None:
        # ``Lp_pool(x, p=1) = avg_pool(|x|) · K = sum(|x|)`` over the window.
        x = lucid.tensor([[[1.0, -2.0, 3.0, -4.0]]])
        out = F.lp_pool1d(x, norm_type=1.0, kernel_size=2)
        np.testing.assert_allclose(out.numpy(), [[[3.0, 7.0]]], atol=1e-6)

    def test_lp_pool_p_zero_rejected(self) -> None:
        x = lucid.arange(0.0, 4.0, 1.0).reshape(1, 1, 4)
        with pytest.raises(ValueError):
            F.lp_pool1d(x, norm_type=0.0, kernel_size=2)


class TestMaxUnpool:
    def test_unpool1d(self) -> None:
        v = lucid.tensor([[[2.0, 4.0]]])
        idx = lucid.tensor([[[1, 3]]], dtype=lucid.int64)
        out = F.max_unpool1d(v, idx, kernel_size=2, output_size=(4,))
        np.testing.assert_array_equal(out.numpy(), [[[0.0, 2.0, 0.0, 4.0]]])

    def test_unpool2d(self) -> None:
        v = lucid.tensor([[[[5.0]]]])
        idx = lucid.tensor([[[[3]]]], dtype=lucid.int64)  # bottom-right of 2x2
        out = F.max_unpool2d(v, idx, kernel_size=2, output_size=(2, 2))
        np.testing.assert_array_equal(out.numpy(), [[[[0.0, 0.0], [0.0, 5.0]]]])

    def test_unpool_requires_output_size(self) -> None:
        v = lucid.tensor([[[1.0]]])
        idx = lucid.tensor([[[0]]], dtype=lucid.int64)
        with pytest.raises(ValueError):
            F.max_unpool1d(v, idx, kernel_size=2)


class TestFractionalMaxPool:
    def test_2d_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            F.fractional_max_pool2d(lucid.zeros(1, 1, 4, 4), kernel_size=2)

    def test_3d_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            F.fractional_max_pool3d(lucid.zeros(1, 1, 4, 4, 4), kernel_size=2)


# ── losses ─────────────────────────────────────────────────────────────────


class TestSoftMarginLoss:
    def test_default_mean(self) -> None:
        # log(1 + exp(0)) = log 2 — when target · input == 0.
        out = F.soft_margin_loss(lucid.tensor([0.0]), lucid.tensor([1.0]))
        assert abs(out.item() - math.log(2.0)) < 1e-6

    def test_reduction_none(self) -> None:
        out = F.soft_margin_loss(
            lucid.tensor([0.0, 0.0]),
            lucid.tensor([1.0, -1.0]),
            reduction="none",
        )
        np.testing.assert_allclose(out.numpy(), [math.log(2.0)] * 2, atol=1e-6)

    def test_invalid_reduction(self) -> None:
        with pytest.raises(ValueError):
            F.soft_margin_loss(
                lucid.tensor([0.0]), lucid.tensor([1.0]), reduction="bogus"
            )


class TestMultilabelSoftMarginLoss:
    def test_zero_input_zero_target_minus_log2(self) -> None:
        # x=0, t=0 → -[t·log0.5 + (1-t)·log0.5] = log 2 per class, mean = log 2.
        out = F.multilabel_soft_margin_loss(
            lucid.tensor([[0.0, 0.0, 0.0]]), lucid.tensor([[0.0, 0.0, 0.0]])
        )
        assert abs(out.item() - math.log(2.0)) < 1e-6

    def test_invalid_reduction(self) -> None:
        with pytest.raises(ValueError):
            F.multilabel_soft_margin_loss(
                lucid.tensor([[0.0]]),
                lucid.tensor([[0.0]]),
                reduction="bogus",
            )


# ── shape / distance ───────────────────────────────────────────────────────


class TestChannelShuffle:
    def test_identity_groups_one(self) -> None:
        x = lucid.arange(0.0, 12.0, 1.0).reshape(1, 6, 2)
        np.testing.assert_array_equal(F.channel_shuffle(x, groups=1).numpy(), x.numpy())

    def test_groups_two(self) -> None:
        # 4 channels split into 2 groups of 2: [0,1,2,3] → [0,2,1,3]
        x = lucid.tensor([[[10.0], [11.0], [12.0], [13.0]]])  # (1, 4, 1)
        out = F.channel_shuffle(x, groups=2).numpy()
        np.testing.assert_array_equal(out, [[[10.0], [12.0], [11.0], [13.0]]])

    def test_indivisible_rejected(self) -> None:
        with pytest.raises(ValueError):
            F.channel_shuffle(lucid.zeros(1, 5, 2), groups=2)


class TestPdist:
    def test_three_points_l2(self) -> None:
        pts = lucid.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        out = F.pdist(pts).numpy()
        np.testing.assert_allclose(out, [1.0, 1.0, math.sqrt(2.0)], atol=1e-5)

    def test_l1(self) -> None:
        pts = lucid.tensor([[0.0, 0.0], [3.0, 4.0]])
        # Manhattan distance = 7
        out = F.pdist(pts, p=1.0).numpy()
        np.testing.assert_allclose(out, [7.0], atol=1e-5)

    def test_single_point_empty(self) -> None:
        out = F.pdist(lucid.tensor([[1.0, 2.0]]))
        assert out.shape == (0,)

    def test_non_2d_rejected(self) -> None:
        with pytest.raises(ValueError):
            F.pdist(lucid.tensor([1.0, 2.0]))
