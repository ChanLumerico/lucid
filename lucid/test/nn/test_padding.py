"""Tests for the padding modules — focuses on the gaps filled in this pass
(``ZeroPad1d`` / ``ZeroPad3d``)."""

import numpy as np

import lucid
import lucid.nn as nn


class TestZeroPad1d:
    def test_pads_with_zeros(self) -> None:
        layer: nn.ZeroPad1d = nn.ZeroPad1d(2)
        x: lucid.Tensor = lucid.tensor([[[1.0, 2.0, 3.0]]])  # (N=1, C=1, L=3)
        np.testing.assert_allclose(
            layer(x).numpy(),
            np.array([[[0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]]]),
        )

    def test_asymmetric_padding(self) -> None:
        layer: nn.ZeroPad1d = nn.ZeroPad1d((1, 3))
        x: lucid.Tensor = lucid.tensor([[[1.0, 2.0]]])
        np.testing.assert_allclose(
            layer(x).numpy(),
            np.array([[[0.0, 1.0, 2.0, 0.0, 0.0, 0.0]]]),
        )


class TestZeroPad3d:
    def test_uniform_padding(self) -> None:
        layer: nn.ZeroPad3d = nn.ZeroPad3d(1)
        x: lucid.Tensor = lucid.tensor(np.ones((1, 1, 2, 2, 2), dtype=np.float32))
        out: np.ndarray = layer(x).numpy()
        assert out.shape == (1, 1, 4, 4, 4)
        # Inner block is all ones; everywhere else is zero.
        assert out[0, 0, 1:3, 1:3, 1:3].sum() == 8.0
        assert out[0, 0, 0, :, :].sum() == 0.0
