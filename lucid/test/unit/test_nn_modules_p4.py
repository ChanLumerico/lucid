"""Unit tests for the P4 ``lucid.nn`` module-class additions."""

import math

import numpy as np
import pytest

import lucid
import lucid.nn as nn

# ── pooling — MaxUnpool / FractionalMaxPool ────────────────────────────────


class TestMaxUnpool:
    def test_unpool1d_module(self) -> None:
        m = nn.MaxUnpool1d(kernel_size=2)
        v = lucid.tensor([[[2.0, 4.0]]])
        idx = lucid.tensor([[[1, 3]]], dtype=lucid.int64)
        out = m(v, idx, output_size=(4,)).numpy()
        np.testing.assert_array_equal(out, [[[0.0, 2.0, 0.0, 4.0]]])

    def test_unpool2d_module(self) -> None:
        m = nn.MaxUnpool2d(kernel_size=2)
        v = lucid.tensor([[[[5.0]]]])
        idx = lucid.tensor([[[[3]]]], dtype=lucid.int64)
        out = m(v, idx, output_size=(2, 2)).numpy()
        np.testing.assert_array_equal(out, [[[[0.0, 0.0], [0.0, 5.0]]]])

    def test_unpool3d_module_runs(self) -> None:
        m = nn.MaxUnpool3d(kernel_size=2)
        v = lucid.tensor([[[[[7.0]]]]])
        idx = lucid.tensor([[[[[0]]]]], dtype=lucid.int64)
        out = m(v, idx, output_size=(2, 2, 2))
        assert out.shape == (1, 1, 2, 2, 2)


class TestFractionalMaxPool:
    def test_2d_not_implemented(self) -> None:
        m = nn.FractionalMaxPool2d(kernel_size=2)
        with pytest.raises(NotImplementedError):
            m(lucid.zeros(1, 1, 4, 4))

    def test_3d_not_implemented(self) -> None:
        m = nn.FractionalMaxPool3d(kernel_size=2)
        with pytest.raises(NotImplementedError):
            m(lucid.zeros(1, 1, 4, 4, 4))


# ── padding — Reflection3d / Circular{1,2,3}d ──────────────────────────────


class TestReflectionPad3d:
    def test_shape(self) -> None:
        m = nn.ReflectionPad3d(1)
        x = lucid.zeros(1, 1, 3, 3, 3)
        assert m(x).shape == (1, 1, 5, 5, 5)


class TestCircularPad:
    def test_pad1d_wraps(self) -> None:
        m = nn.CircularPad1d(1)
        x = lucid.tensor([[[1.0, 2.0, 3.0]]])
        # Wraps: last element → front, first → back.
        np.testing.assert_array_equal(m(x).numpy(), [[[3.0, 1.0, 2.0, 3.0, 1.0]]])

    def test_pad2d_shape(self) -> None:
        m = nn.CircularPad2d((1, 1, 1, 1))
        x = lucid.zeros(1, 1, 3, 3)
        assert m(x).shape == (1, 1, 5, 5)

    def test_pad3d_shape(self) -> None:
        m = nn.CircularPad3d(1)
        x = lucid.zeros(1, 1, 3, 3, 3)
        assert m(x).shape == (1, 1, 5, 5, 5)


# ── loss — Soft / MultiLabelSoft / TripletMarginWithDistance ───────────────


class TestSoftMarginLoss:
    def test_default_mean(self) -> None:
        m = nn.SoftMarginLoss()
        assert (
            abs(m(lucid.tensor([0.0]), lucid.tensor([1.0])).item() - math.log(2.0))
            < 1e-6
        )


class TestMultiLabelSoftMarginLoss:
    def test_default_mean(self) -> None:
        m = nn.MultiLabelSoftMarginLoss()
        assert (
            abs(
                m(
                    lucid.tensor([[0.0, 0.0, 0.0]]),
                    lucid.tensor([[0.0, 0.0, 0.0]]),
                ).item()
                - math.log(2.0)
            )
            < 1e-6
        )


class TestTripletMarginWithDistanceLoss:
    def test_zero_when_d_neg_minus_d_pos_exceeds_margin(self) -> None:
        # anchor==positive ⇒ d_pos=0, d_neg=√2 ≈ 1.414 > margin=1 ⇒ loss=0.
        m = nn.TripletMarginWithDistanceLoss(margin=1.0)
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[1.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        assert abs(m(a, p, n).item()) < 1e-6

    def test_custom_distance(self) -> None:
        # L1 distance — ‖a-p‖₁ = 0, ‖a-n‖₁ = 2, margin=1, swap=False.
        # loss = max(0 - 2 + 1, 0) = 0.
        m = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: (x - y).abs().sum(dim=-1),
            margin=1.0,
        )
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[1.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        assert abs(m(a, p, n).item()) < 1e-6

    def test_invalid_reduction(self) -> None:
        m = nn.TripletMarginWithDistanceLoss(reduction="bogus")
        a = lucid.tensor([[1.0, 0.0]])
        with pytest.raises(ValueError):
            m(a, a, a)


# ── misc — ChannelShuffle ──────────────────────────────────────────────────


class TestChannelShuffle:
    def test_groups_two(self) -> None:
        m = nn.ChannelShuffle(groups=2)
        x = lucid.tensor([[[10.0], [11.0], [12.0], [13.0]]])
        out = m(x).numpy()
        np.testing.assert_array_equal(out, [[[10.0], [12.0], [11.0], [13.0]]])

    def test_indivisible_rejected(self) -> None:
        m = nn.ChannelShuffle(groups=2)
        with pytest.raises(ValueError):
            m(lucid.zeros(1, 5, 2))


# ── public surface check ───────────────────────────────────────────────────


class TestSurface:
    def test_modules_visible_under_lucid_nn(self) -> None:
        # Every P4 addition should appear at the documented public path.
        for name in (
            "MaxUnpool1d",
            "MaxUnpool2d",
            "MaxUnpool3d",
            "FractionalMaxPool2d",
            "FractionalMaxPool3d",
            "ReflectionPad3d",
            "CircularPad1d",
            "CircularPad2d",
            "CircularPad3d",
            "ChannelShuffle",
            "SoftMarginLoss",
            "MultiLabelSoftMarginLoss",
            "TripletMarginWithDistanceLoss",
        ):
            assert hasattr(nn, name), f"nn.{name} missing"
