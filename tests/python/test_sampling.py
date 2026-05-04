"""
Tests for lucid.nn.functional.sampling — interpolate, grid_sample, affine_grid,
unfold, pad.  Each test probes both output shape and basic numerical properties.
"""

import numpy as np
import pytest
import lucid
import lucid.nn.functional as F


# ── interpolate ───────────────────────────────────────────────────────────────


class TestInterpolate:
    def test_nearest_upsample_2d_size(self):
        x = lucid.randn(2, 3, 4, 4)
        y = F.interpolate(x, size=(8, 8), mode="nearest")
        assert y.shape == (2, 3, 8, 8)

    def test_nearest_downsample_2d(self):
        x = lucid.randn(1, 1, 8, 8)
        y = F.interpolate(x, size=(4, 4), mode="nearest")
        assert y.shape == (1, 1, 4, 4)

    def test_nearest_scale_factor_2x(self):
        x = lucid.randn(2, 3, 4, 4)
        y = F.interpolate(x, scale_factor=2.0, mode="nearest")
        assert y.shape == (2, 3, 8, 8)

    def test_nearest_scale_factor_half(self):
        x = lucid.randn(1, 2, 8, 8)
        y = F.interpolate(x, scale_factor=0.5, mode="nearest")
        assert y.shape == (1, 2, 4, 4)

    def test_nearest_identity(self):
        x = lucid.randn(1, 1, 4, 4)
        y = F.interpolate(x, size=(4, 4), mode="nearest")
        np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-5)

    def test_bilinear_upsample(self):
        x = lucid.randn(2, 3, 4, 4)
        y = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)
        assert y.shape == (2, 3, 8, 8)

    def test_bilinear_align_corners(self):
        x = lucid.randn(1, 1, 4, 4)
        y = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=True)
        assert y.shape == (1, 1, 8, 8)

    def test_bilinear_nonsquare(self):
        x = lucid.randn(1, 3, 4, 8)
        y = F.interpolate(x, size=(6, 12), mode="bilinear", align_corners=False)
        assert y.shape == (1, 3, 6, 12)

    def test_nearest_3d(self):
        x = lucid.randn(1, 2, 2, 4, 4)
        y = F.interpolate(x, size=(4, 8, 8), mode="nearest")
        assert y.shape == (1, 2, 4, 8, 8)

    def test_unsupported_mode_raises(self):
        x = lucid.randn(1, 1, 4, 4)
        with pytest.raises(ValueError, match="Unsupported"):
            F.interpolate(x, size=(8, 8), mode="lanczos")


# ── grid_sample ───────────────────────────────────────────────────────────────


class TestGridSample:
    def test_identity_grid(self):
        x = lucid.randn(2, 3, 8, 8)
        # Identity grid: coords in [-1, 1] range mapped to themselves
        grid = lucid.zeros(2, 8, 8, 2)
        y = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
        assert y.shape == (2, 3, 8, 8)

    def test_output_size_from_grid(self):
        x = lucid.randn(1, 4, 16, 16)
        grid = lucid.zeros(1, 8, 12, 2)  # output (N, H_out, W_out, 2)
        y = F.grid_sample(x, grid)
        assert y.shape == (1, 4, 8, 12)

    def test_grid_sample_zeros_output(self):
        # Grid with all coords at extreme (-2, -2) → out of bounds → should clamp to border
        x = lucid.ones(1, 1, 4, 4)
        grid = lucid.zeros(1, 4, 4, 2)  # center coords → all ones
        y = F.grid_sample(x, grid)
        assert y.shape == (1, 1, 4, 4)


# ── affine_grid ───────────────────────────────────────────────────────────────


class TestAffineGrid:
    def test_identity_theta(self):
        # Identity transformation: [[1,0,0],[0,1,0]]
        theta = lucid.tensor(np.tile(
            np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (2, 1, 1)
        ).astype(np.float32))
        grid = F.affine_grid(theta, [2, 3, 8, 8])
        assert grid.shape == (2, 8, 8, 2)

    def test_grid_values_range(self):
        theta = lucid.tensor(np.tile(
            np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (1, 1, 1)
        ).astype(np.float32))
        grid = F.affine_grid(theta, [1, 1, 4, 4])
        arr = grid.numpy()
        # Grid coordinates should be in [-1, 1] range for identity transform
        assert arr.min() >= -1.1 and arr.max() <= 1.1

    def test_scale_theta(self):
        # Scaling by 0.5: zoom in
        theta = lucid.tensor(np.tile(
            np.array([[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]]), (1, 1, 1)
        ).astype(np.float32))
        grid = F.affine_grid(theta, [1, 1, 8, 8])
        assert grid.shape == (1, 8, 8, 2)

    def test_affine_grid_then_grid_sample(self):
        theta = lucid.tensor(np.tile(
            np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]), (2, 1, 1)
        ).astype(np.float32))
        x = lucid.randn(2, 3, 8, 8)
        grid = F.affine_grid(theta, [2, 3, 8, 8])
        y = F.grid_sample(x, grid)
        assert y.shape == (2, 3, 8, 8)


# ── unfold ────────────────────────────────────────────────────────────────────


class TestUnfold:
    def test_unfold_shape_basic(self):
        # (N, C, H, W) → (N, C*kH*kW, L)
        x = lucid.randn(2, 3, 8, 8)
        y = F.unfold(x, kernel_size=2, stride=2)
        # L = (8-2)//2+1 * (8-2)//2+1 = 4*4 = 16
        assert y.shape == (2, 3 * 2 * 2, 16)

    def test_unfold_shape_with_padding(self):
        x = lucid.randn(1, 2, 4, 4)
        y = F.unfold(x, kernel_size=3, padding=1)
        # L = (4+2-3)//1+1 = 4*4 = 16
        assert y.shape == (1, 2 * 3 * 3, 16)

    def test_unfold_kernel_1x1(self):
        x = lucid.randn(2, 4, 6, 6)
        y = F.unfold(x, kernel_size=1)
        # With kernel 1x1, output is (N, C, H*W)
        assert y.shape == (2, 4, 36)

    def test_unfold_matches_expected_values(self):
        x = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
        y = F.unfold(x, kernel_size=2, stride=1)
        # Single window covers entire 2x2 — one location
        assert y.shape == (1, 4, 1)
        expected = np.array([[[1.0], [2.0], [3.0], [4.0]]])
        np.testing.assert_allclose(y.numpy(), expected, atol=1e-5)


# ── pad ───────────────────────────────────────────────────────────────────────


class TestPad:
    def test_pad_last_dim(self):
        x = lucid.randn(2, 3, 4)
        y = F.pad(x, (1, 1))  # pad last dim
        assert y.shape == (2, 3, 6)

    def test_pad_last_two_dims(self):
        x = lucid.randn(2, 3, 4, 4)
        y = F.pad(x, (1, 1, 2, 2))  # last dim +2, second-to-last +4
        assert y.shape == (2, 3, 8, 6)

    def test_pad_zeros(self):
        x = lucid.tensor([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
        y = F.pad(x, (1, 1), value=0.0)
        assert y.shape == (1, 1, 5)
        expected = np.array([[[0.0, 1.0, 2.0, 3.0, 0.0]]])
        np.testing.assert_allclose(y.numpy(), expected, atol=1e-5)

    def test_pad_constant_value(self):
        x = lucid.ones(1, 1, 3)
        y = F.pad(x, (2, 2), value=5.0)
        assert y.shape == (1, 1, 7)
        arr = y.numpy()
        np.testing.assert_allclose(arr[0, 0, :2], [5.0, 5.0], atol=1e-5)
        np.testing.assert_allclose(arr[0, 0, 2:5], [1.0, 1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(arr[0, 0, 5:], [5.0, 5.0], atol=1e-5)

    def test_pad_4d_asymmetric(self):
        x = lucid.randn(1, 1, 3, 3)
        y = F.pad(x, (0, 1, 0, 1))  # right=1, bottom=1
        assert y.shape == (1, 1, 4, 4)

    def test_pad_zero_padding_identity(self):
        x = lucid.randn(2, 3, 4)
        y = F.pad(x, (0, 0))
        np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-6)
