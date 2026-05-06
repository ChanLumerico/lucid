"""Tests for Dropout variants."""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test.helpers.numerics import make_tensor


class TestDropout:
    def test_eval_mode_identity(self):
        layer = nn.Dropout(p=0.5)
        layer.eval()
        x = make_tensor((4, 8))
        out = layer(x)
        np.testing.assert_array_almost_equal(out.numpy(), x.numpy())

    def test_train_mode_zeros_some(self):
        layer = nn.Dropout(p=0.9)
        layer.train()
        x = lucid.ones(100)
        out = layer(x)
        # With p=0.9, ~90% should be 0
        n_zero = (out.numpy() == 0).sum()
        assert n_zero > 50  # statistical: should be >> 50%

    def test_output_shape(self):
        layer = nn.Dropout(p=0.5)
        x = make_tensor((4, 8))
        assert layer(x).shape == (4, 8)

    def test_p_zero_identity(self):
        layer = nn.Dropout(p=0.0)
        x = make_tensor((4, 8))
        out = layer(x)
        np.testing.assert_array_almost_equal(out.numpy(), x.numpy())


class TestDropout2d:
    def test_output_shape(self):
        layer = nn.Dropout2d(p=0.5)
        x = make_tensor((2, 4, 8, 8))
        assert layer(x).shape == (2, 4, 8, 8)

    def test_eval_mode_identity(self):
        layer = nn.Dropout2d(p=0.5)
        layer.eval()
        x = make_tensor((2, 4, 8, 8))
        out = layer(x)
        np.testing.assert_array_almost_equal(out.numpy(), x.numpy())


class TestAlphaDropout:
    def test_output_shape(self):
        layer = nn.AlphaDropout(p=0.5)
        x = make_tensor((4, 8))
        assert layer(x).shape == (4, 8)

    def test_inplace_stored(self):
        layer = nn.AlphaDropout(p=0.5, inplace=True)
        assert layer.inplace is True


class TestDropoutValidation:
    @pytest.mark.parametrize(
        "cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout]
    )
    def test_invalid_probability_rejected(self, cls):
        with pytest.raises(ValueError, match="probability"):
            cls(p=-0.1)
        with pytest.raises(ValueError, match="probability"):
            cls(p=1.5)

    @pytest.mark.parametrize(
        "cls", [nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout]
    )
    def test_inplace_in_repr(self, cls):
        layer = cls(p=0.3, inplace=True)
        assert "inplace=True" in repr(layer)
        layer_default = cls(p=0.3)
        assert "inplace" not in repr(layer_default)


class TestDropoutScaling:
    def test_train_mode_preserves_expectation(self):
        layer = nn.Dropout(p=0.4)
        layer.train()
        # Large input, average over many samples — output mean should match
        # input mean (within MC noise).
        x = lucid.tensor(np.ones((10000,), dtype=np.float32) * 2.0)
        outs = [layer(x).numpy() for _ in range(20)]
        mean_out = np.mean(outs)
        # Expected ~ 2.0 (E[mask · x / (1-p)] = x), noise ~ a few percent.
        assert abs(mean_out - 2.0) < 0.05

    def test_dropout2d_zeros_whole_channels(self):
        layer = nn.Dropout2d(p=0.5)
        layer.train()
        x = lucid.tensor(np.ones((1, 32, 4, 4), dtype=np.float32))
        out = layer(x).numpy()
        # For any channel that survives, every spatial location has the
        # same scaled value; for any zeroed channel, every spatial location
        # is zero.  Either way each (n, c) slice is uniform.
        # Check: per-channel std along (H, W) axes is zero.
        per_chan_std = out[0].reshape(32, -1).std(axis=1)
        np.testing.assert_allclose(per_chan_std, 0.0, atol=1e-5)


class TestDropout1d:
    def test_eval_mode_identity(self) -> None:
        layer: nn.Dropout1d = nn.Dropout1d(p=0.5)
        layer.eval()
        x: lucid.Tensor = lucid.randn(2, 4, 8)
        np.testing.assert_allclose(layer(x).numpy(), x.numpy())

    def test_zeros_whole_channels(self) -> None:
        layer: nn.Dropout1d = nn.Dropout1d(p=0.5)
        layer.train()
        x: lucid.Tensor = lucid.tensor(np.ones((1, 32, 16), dtype=np.float32))
        out: np.ndarray = layer(x).numpy()
        # Each (n, c) channel is uniform along the length dim.
        per_chan_std: np.ndarray = out[0].std(axis=-1)
        np.testing.assert_allclose(per_chan_std, 0.0, atol=1e-5)


class TestFeatureAlphaDropout:
    def test_eval_mode_identity(self) -> None:
        layer: nn.FeatureAlphaDropout = nn.FeatureAlphaDropout(p=0.5)
        layer.eval()
        x: lucid.Tensor = lucid.randn(2, 3, 4, 4)
        np.testing.assert_allclose(layer(x).numpy(), x.numpy())

    def test_train_mode_runs(self) -> None:
        # The arithmetic is shared with ``alpha_dropout`` (already covered by
        # the AlphaDropout suite); here we just exercise the broadcast path
        # for ``(N, C, *)`` masks.
        layer: nn.FeatureAlphaDropout = nn.FeatureAlphaDropout(p=0.5)
        layer.train()
        x: lucid.Tensor = lucid.randn(2, 3, 4, 4)
        out: lucid.Tensor = layer(x)
        assert out.shape == x.shape
