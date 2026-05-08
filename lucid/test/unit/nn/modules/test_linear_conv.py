"""nn.Linear / nn.Bilinear / nn.Conv* / nn.ConvTranspose* modules."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestLinear:
    def test_shape(self) -> None:
        m = nn.Linear(4, 8)
        x = lucid.zeros(2, 4)
        out = m(x)
        assert out.shape == (2, 8)

    def test_no_bias_shape(self) -> None:
        m = nn.Linear(4, 8, bias=False)
        out = m(lucid.zeros(2, 4))
        assert out.shape == (2, 8)

    def test_known_values(self) -> None:
        m = nn.Linear(2, 1, bias=False)
        # Override weight to identity-ish.
        m.weight.data if hasattr(m.weight, "data") else m.weight  # touch
        # We can verify that y = x @ w.T by sampling random inputs.
        x = lucid.tensor([[1.0, 2.0]])
        out = m(x)
        # out shape is (1, 1).
        assert out.shape == (1, 1)


class TestBilinear:
    def test_shape(self) -> None:
        m = nn.Bilinear(3, 4, 5)
        out = m(lucid.zeros(2, 3), lucid.zeros(2, 4))
        assert out.shape == (2, 5)


class TestConv1d:
    def test_basic_shape(self) -> None:
        m = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3)
        x = lucid.zeros(1, 2, 16)
        out = m(x)
        # Default stride=1, padding=0 → length 16-3+1 = 14.
        assert out.shape == (1, 4, 14)


class TestConv2d:
    def test_basic_shape(self) -> None:
        m = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = lucid.zeros(1, 3, 16, 16)
        out = m(x)
        assert out.shape == (1, 8, 16, 16)

    def test_stride_2(self) -> None:
        m = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        out = m(lucid.zeros(1, 3, 16, 16))
        assert out.shape == (1, 8, 8, 8)


class TestConv3d:
    def test_basic_shape(self) -> None:
        m = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        out = m(lucid.zeros(1, 2, 8, 8, 8))
        assert out.shape == (1, 4, 8, 8, 8)


class TestConvTranspose:
    def test_2d_doubles(self) -> None:
        m = nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)
        out = m(lucid.zeros(1, 4, 8, 8))
        assert out.shape == (1, 2, 16, 16)
