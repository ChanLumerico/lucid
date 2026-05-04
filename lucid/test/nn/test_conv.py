"""Tests for Conv1d/2d/3d + ConvTranspose."""

import pytest
import math
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


def _conv_out(i, k, s, p, d=1):
    return math.floor((i + 2 * p - d * (k - 1) - 1) / s) + 1


class TestConv1d:
    def test_output_shape_basic(self):
        x = make_tensor((2, 4, 16))
        layer = nn.Conv1d(4, 8, kernel_size=3)
        out = layer(x)
        expected_l = _conv_out(16, 3, 1, 0)
        assert out.shape == (2, 8, expected_l)

    def test_output_shape_with_padding(self):
        x = make_tensor((2, 4, 16))
        layer = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        out = layer(x)
        assert out.shape == (2, 8, 16)

    def test_output_shape_stride2(self):
        x = make_tensor((2, 4, 16))
        layer = nn.Conv1d(4, 8, kernel_size=3, stride=2, padding=1)
        out = layer(x)
        expected_l = _conv_out(16, 3, 2, 1)
        assert out.shape == (2, 8, expected_l)

    def test_bias_shape(self):
        layer = nn.Conv1d(4, 8, 3)
        assert layer.bias.shape == (8,)

    def test_no_bias(self):
        layer = nn.Conv1d(4, 8, 3, bias=False)
        assert layer.bias is None


class TestConv2d:
    def test_output_shape_basic(self):
        x = make_tensor((2, 3, 8, 8))
        layer = nn.Conv2d(3, 16, kernel_size=3)
        out = layer(x)
        expected = _conv_out(8, 3, 1, 0)
        assert out.shape == (2, 16, expected, expected)

    def test_same_padding(self):
        x = make_tensor((2, 3, 8, 8))
        layer = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        out = layer(x)
        assert out.shape == (2, 16, 8, 8)

    def test_stride2(self):
        x = make_tensor((2, 3, 8, 8))
        layer = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        out = layer(x)
        assert out.shape == (2, 16, 4, 4)

    def test_groups(self):
        x = make_tensor((2, 4, 8, 8))
        layer = nn.Conv2d(4, 8, kernel_size=3, groups=2, padding=1)
        out = layer(x)
        assert out.shape == (2, 8, 8, 8)

    def test_weight_shape(self):
        layer = nn.Conv2d(3, 16, kernel_size=3)
        assert layer.weight.shape == (16, 3, 3, 3)

    def test_backward_x_grad(self):
        layer = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        x = make_tensor((2, 3, 8, 8), requires_grad=True)
        out = layer(x)
        lucid.sum(out).backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 3, 8, 8)


class TestConvTranspose:
    def test_conv_transpose1d_shape(self):
        x = make_tensor((2, 4, 8))
        layer = nn.ConvTranspose1d(4, 2, kernel_size=3)
        out = layer(x)
        # output_size = (8 - 1) * 1 - 0 + 3 = 10
        assert out.shape == (2, 2, 10)

    def test_conv_transpose2d_shape(self):
        x = make_tensor((2, 4, 4, 4))
        layer = nn.ConvTranspose2d(4, 2, kernel_size=3, stride=2, padding=1)
        out = layer(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 2

    def test_conv_transpose_no_bias(self):
        layer = nn.ConvTranspose1d(4, 2, 3, bias=False)
        assert layer.bias is None
