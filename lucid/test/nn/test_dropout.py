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
