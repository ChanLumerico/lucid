"""Tests for the upsampling modules added in this pass."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestUpsamplingNearest2d:
    def test_doubles_spatial_dims(self) -> None:
        layer: nn.UpsamplingNearest2d = nn.UpsamplingNearest2d(scale_factor=2)
        x: lucid.Tensor = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        out: np.ndarray = layer(x).numpy()
        assert out.shape == (1, 1, 4, 4)
        # Each input cell is replicated into a 2×2 block.
        np.testing.assert_allclose(
            out[0, 0],
            np.array(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0],
                ]
            ),
        )

    def test_size_target(self) -> None:
        layer: nn.UpsamplingNearest2d = nn.UpsamplingNearest2d(size=(6, 6))
        x: lucid.Tensor = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        assert layer(x).shape == (1, 1, 6, 6)


class TestUpsamplingBilinear2d:
    def test_align_corners_default_true(self) -> None:
        # ``UpsamplingBilinear2d`` pre-sets ``align_corners=True`` — the
        # corners of the input land exactly at the corners of the output.
        layer: nn.UpsamplingBilinear2d = nn.UpsamplingBilinear2d(scale_factor=2)
        x: lucid.Tensor = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        out: np.ndarray = layer(x).numpy()
        assert out.shape == (1, 1, 4, 4)
        # Corners preserved exactly.
        assert out[0, 0, 0, 0] == pytest.approx(1.0)
        assert out[0, 0, 0, -1] == pytest.approx(2.0)
        assert out[0, 0, -1, 0] == pytest.approx(3.0)
        assert out[0, 0, -1, -1] == pytest.approx(4.0)
