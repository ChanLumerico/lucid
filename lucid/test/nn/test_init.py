"""Tests for nn.init initialisers."""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.init as init


# ── trunc_normal_ ────────────────────────────────────────────────────────────


class TestTruncNormal:
    def test_values_within_bounds(self):
        t = lucid.zeros(200, 50)
        init.trunc_normal_(t, mean=0.0, std=1.0, a=-1.5, b=1.5)
        arr = t.numpy()
        assert arr.min() >= -1.5 - 1e-5
        assert arr.max() <= 1.5 + 1e-5
        # Mean should be near 0 for symmetric truncation.
        assert abs(arr.mean()) < 0.05


# ── orthogonal_ ──────────────────────────────────────────────────────────────


class TestOrthogonal:
    def test_orthogonal_square(self):
        t = lucid.zeros(8, 8)
        init.orthogonal_(t)
        arr = t.numpy()
        # Q Q^T ≈ I.
        np.testing.assert_allclose(arr @ arr.T, np.eye(8), atol=1e-4)

    def test_orthogonal_tall(self):
        t = lucid.zeros(12, 4)
        init.orthogonal_(t)
        arr = t.numpy()
        # Columns are orthonormal: Q^T Q ≈ I.
        np.testing.assert_allclose(arr.T @ arr, np.eye(4), atol=1e-4)


# ── sparse_ ──────────────────────────────────────────────────────────────────


class TestSparse:
    def test_per_column_zero_count(self):
        rows, cols = 20, 6
        sparsity = 0.4
        t = lucid.zeros(rows, cols)
        init.sparse_(t, sparsity=sparsity)
        arr = t.numpy()
        expected_zeros = int(math.floor(sparsity * rows))
        for c in range(cols):
            actual = int((arr[:, c] == 0.0).sum())
            assert actual == expected_zeros, (
                f"column {c}: {actual} zeros, expected {expected_zeros}"
            )

    def test_non_zero_entries_have_finite_std(self):
        t = lucid.zeros(100, 4)
        init.sparse_(t, sparsity=0.3, std=0.05)
        arr = t.numpy()
        nz = arr[arr != 0.0]
        # Standard deviation should be near 0.05 (Monte-Carlo noise allowed).
        assert 0.02 < nz.std() < 0.08

    def test_non_2d_rejected(self):
        with pytest.raises(ValueError, match="2D"):
            init.sparse_(lucid.zeros(5, 6, 7), sparsity=0.3)

    def test_invalid_sparsity_rejected(self):
        with pytest.raises(ValueError, match="sparsity"):
            init.sparse_(lucid.zeros(8, 8), sparsity=1.5)
        with pytest.raises(ValueError, match="sparsity"):
            init.sparse_(lucid.zeros(8, 8), sparsity=-0.1)


# ── dirac_ ───────────────────────────────────────────────────────────────────


class TestDirac:
    def test_dirac_2d_kernel(self):
        # (Cout=4, Cin/g=4, kH=3, kW=3) — square identity, kernel centre.
        t = lucid.zeros(4, 4, 3, 3)
        init.dirac_(t, groups=1)
        arr = t.numpy()
        # Centre of each (i, i) slice is 1, everything else is 0.
        for i in range(4):
            for j in range(4):
                if i == j:
                    assert arr[i, j, 1, 1] == 1.0
                else:
                    assert arr[i, j, 1, 1] == 0.0
        # Off-centre positions are zero everywhere.
        assert (arr[:, :, 0, 0] == 0).all()
        assert (arr[:, :, 2, 2] == 0).all()

    def test_dirac_grouped(self):
        # Cout=8 with groups=2 means each group sees 4 out_channels paired
        # with 4 in_channels — diag of each group's slab is 1.
        t = lucid.zeros(8, 4, 3, 3)
        init.dirac_(t, groups=2)
        arr = t.numpy()
        for g in range(2):
            for d in range(4):
                out_idx = g * 4 + d
                assert arr[out_idx, d, 1, 1] == 1.0

    def test_dirac_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match="3/4/5"):
            init.dirac_(lucid.zeros(4, 8))

    def test_dirac_groups_validation(self):
        # 5 out_channels not divisible by 2 groups.
        with pytest.raises(ValueError, match="divisible"):
            init.dirac_(lucid.zeros(5, 4, 3, 3), groups=2)


# ── kaiming / xavier / calculate_gain ────────────────────────────────────────


class TestGainAndFan:
    def test_calculate_gain_known_nonlinearities(self):
        assert init.calculate_gain("linear") == 1.0
        assert init.calculate_gain("sigmoid") == 1.0
        assert math.isclose(init.calculate_gain("tanh"), 5.0 / 3.0)
        assert math.isclose(init.calculate_gain("relu"), math.sqrt(2.0))

    def test_calculate_gain_leaky_relu_slope(self):
        # Default slope 0.01 → sqrt(2 / (1 + 0.01²)).
        expected = math.sqrt(2.0 / (1 + 0.01**2))
        assert math.isclose(init.calculate_gain("leaky_relu"), expected)
        # Custom slope.
        assert math.isclose(
            init.calculate_gain("leaky_relu", 0.2),
            math.sqrt(2.0 / (1 + 0.04)),
        )

    def test_calculate_gain_unsupported_rejects(self):
        with pytest.raises(ValueError, match="Unsupported nonlinearity"):
            init.calculate_gain("swish")

    def test_kaiming_mode_validation(self):
        # An unknown mode raises through _calculate_correct_fan.
        t = lucid.zeros(8, 16)
        with pytest.raises(ValueError, match="Unknown mode"):
            init.kaiming_uniform_(t, mode="bogus")
