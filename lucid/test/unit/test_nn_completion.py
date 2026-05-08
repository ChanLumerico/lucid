"""Unit tests for the final nn / nn.functional gap closure:
``dropout1d``, ``rrelu`` / ``RReLU``, ``lp_pool3d`` / ``LPPool3d``,
``gumbel_softmax``, ``triplet_margin_with_distance_loss``,
``Softmax2d``, ``CosineSimilarity``, ``PairwiseDistance``, and the
``MultiLabelMarginLoss`` casing alias.
"""

import math

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


# ── F.dropout1d ───────────────────────────────────────────────────────────


class TestDropout1d:
    def test_shape_preserved(self) -> None:
        x = lucid.ones(2, 4, 8)
        out = F.dropout1d(x, p=0.5, training=True)
        assert out.shape == (2, 4, 8)

    def test_eval_mode_passthrough(self) -> None:
        x = lucid.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        np.testing.assert_array_equal(
            F.dropout1d(x, p=0.5, training=False).numpy(), x.numpy()
        )


# ── rrelu / RReLU ────────────────────────────────────────────────────────


class TestRrelu:
    def test_eval_uses_midpoint_slope(self) -> None:
        # eval = (lower + upper) / 2 = (1/8 + 1/3) / 2 = 11/48 ≈ 0.2292.
        x = lucid.tensor([-2.0, -1.0, 0.0, 2.0])
        out = F.rrelu(x, training=False).numpy()
        mid = (1.0 / 8.0 + 1.0 / 3.0) / 2.0
        np.testing.assert_allclose(out, [-2.0 * mid, -mid, 0.0, 2.0], atol=1e-5)

    def test_training_slopes_in_range(self) -> None:
        x = lucid.full((1000,), -1.0)
        out = F.rrelu(x, lower=0.1, upper=0.5, training=True).numpy()
        # Each negative gets its own slope ∈ [0.1, 0.5], so out ∈ [-0.5, -0.1].
        assert (out >= -0.5 - 1e-6).all()
        assert (out <= -0.1 + 1e-6).all()

    def test_module_eval_consistent(self) -> None:
        m = nn.RReLU()
        m.eval()
        x = lucid.tensor([-2.0, 1.0])
        np.testing.assert_allclose(
            m(x).numpy(), F.rrelu(x, training=False).numpy(), atol=1e-6
        )


# ── lp_pool3d / LPPool3d ─────────────────────────────────────────────────


class TestLpPool3d:
    def test_shape(self) -> None:
        x = lucid.arange(0.0, 64.0, 1.0).reshape(1, 1, 4, 4, 4)
        out = F.lp_pool3d(x, norm_type=2.0, kernel_size=2)
        assert out.shape == (1, 1, 2, 2, 2)

    def test_p1_equals_sum_abs(self) -> None:
        # p=1 reduces to sum-abs over the window.
        x = lucid.tensor([[[[[1.0, -2.0], [3.0, -4.0]],
                            [[5.0, -6.0], [7.0, -8.0]]]]])
        out = F.lp_pool3d(x, norm_type=1.0, kernel_size=2)
        # Window is the whole 2x2x2 → sum(|x|) = 1+2+3+4+5+6+7+8 = 36.
        np.testing.assert_allclose(out.numpy(), [[[[[36.0]]]]], atol=1e-5)

    def test_module_matches_functional(self) -> None:
        x = lucid.arange(0.0, 64.0, 1.0).reshape(1, 1, 4, 4, 4)
        m = nn.LPPool3d(norm_type=2.0, kernel_size=2)
        np.testing.assert_allclose(
            m(x).numpy(),
            F.lp_pool3d(x, norm_type=2.0, kernel_size=2).numpy(),
            atol=1e-5,
        )


# ── gumbel_softmax ────────────────────────────────────────────────────────


class TestGumbelSoftmax:
    def test_soft_sums_to_one(self) -> None:
        logits = lucid.tensor([[1.0, 2.0, 3.0, 0.5]])
        soft = F.gumbel_softmax(logits, tau=1.0, hard=False).numpy()
        assert abs(soft.sum() - 1.0) < 1e-5

    def test_hard_is_one_hot(self) -> None:
        logits = lucid.tensor([[1.0, 2.0, 3.0, 0.5]])
        hard = F.gumbel_softmax(logits, tau=1.0, hard=True).numpy()
        # Exactly one entry = 1.0, rest = 0.
        assert hard.sum() == 1.0
        assert ((hard == 0.0) | (hard == 1.0)).all()

    def test_low_tau_concentrates_mass(self) -> None:
        # With tiny τ, the soft output approximates the one-hot of argmax.
        lucid.manual_seed(0)
        logits = lucid.tensor([[1.0, 5.0, 1.0]])
        soft = F.gumbel_softmax(logits, tau=0.01, hard=False).numpy()
        assert int(soft.argmax()) == 1
        assert soft[0, 1] > 0.95


# ── triplet_margin_with_distance_loss ────────────────────────────────────


class TestTripletWithDistance:
    def test_default_l2_zero_when_swap_dominates(self) -> None:
        # d_pos = 0, d_neg = √2 > margin → loss = 0.
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[1.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        out = F.triplet_margin_with_distance_loss(a, p, n, margin=1.0)
        assert abs(out.item()) < 1e-6

    def test_custom_distance(self) -> None:
        # L1 distance: d_pos = 0, d_neg = 2; loss = max(0 - 2 + 1, 0) = 0.
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[1.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        out = F.triplet_margin_with_distance_loss(
            a, p, n,
            distance_function=lambda x, y: (x - y).abs().sum(dim=-1),
            margin=1.0,
        )
        assert abs(out.item()) < 1e-6

    def test_module_delegates_to_functional(self) -> None:
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[2.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        m = nn.TripletMarginWithDistanceLoss(margin=2.0)
        np.testing.assert_allclose(
            m(a, p, n).item(),
            F.triplet_margin_with_distance_loss(a, p, n, margin=2.0).item(),
            atol=1e-6,
        )


# ── Softmax2d ─────────────────────────────────────────────────────────────


class TestSoftmax2d:
    def test_sums_to_one_along_channel(self) -> None:
        m = nn.Softmax2d()
        x = lucid.arange(0.0, 24.0, 1.0).reshape(1, 2, 3, 4)
        out = m(x).numpy()
        np.testing.assert_allclose(
            out.sum(axis=1), np.ones((1, 3, 4)), atol=1e-5
        )

    def test_rejects_non_4d(self) -> None:
        m = nn.Softmax2d()
        with pytest.raises(ValueError):
            m(lucid.zeros(2, 3))


# ── CosineSimilarity / PairwiseDistance ──────────────────────────────────


class TestDistanceModules:
    def test_cosine_similarity_orthogonal_zero(self) -> None:
        cs = nn.CosineSimilarity(dim=1)
        a = lucid.tensor([[1.0, 0.0]])
        b = lucid.tensor([[0.0, 1.0]])
        assert abs(cs(a, b).item()) < 1e-5

    def test_cosine_similarity_parallel_one(self) -> None:
        cs = nn.CosineSimilarity(dim=1)
        a = lucid.tensor([[1.0, 2.0]])
        b = lucid.tensor([[2.0, 4.0]])
        assert abs(cs(a, b).item() - 1.0) < 1e-5

    def test_pairwise_distance_l2(self) -> None:
        pd = nn.PairwiseDistance(p=2.0)
        a = lucid.tensor([[0.0, 0.0]])
        b = lucid.tensor([[3.0, 4.0]])
        assert abs(pd(a, b).item() - 5.0) < 1e-4


# ── MultiLabelMarginLoss casing alias ─────────────────────────────────────


class TestMultiLabelMarginLossAlias:
    def test_subclass_relationship(self) -> None:
        # CamelCase alias is a subclass so isinstance checks both ways.
        assert issubclass(nn.MultiLabelMarginLoss, nn.MultilabelMarginLoss)
        m = nn.MultiLabelMarginLoss()
        assert isinstance(m, nn.MultilabelMarginLoss)


# ── Public surface guard ──────────────────────────────────────────────────


class TestSurface:
    def test_F_exposes_all_new(self) -> None:
        for n in (
            "dropout1d", "rrelu", "lp_pool3d",
            "gumbel_softmax", "triplet_margin_with_distance_loss",
        ):
            assert hasattr(F, n), f"F.{n} missing"

    def test_nn_exposes_all_new(self) -> None:
        for n in (
            "RReLU", "LPPool3d", "Softmax2d",
            "CosineSimilarity", "PairwiseDistance",
            "MultiLabelMarginLoss",
        ):
            assert hasattr(nn, n), f"nn.{n} missing"
