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

    def test_conv_transpose_rejects_string_padding(self):
        with pytest.raises(ValueError, match="string padding"):
            nn.ConvTranspose2d(4, 2, 3, padding="same")
        with pytest.raises(ValueError, match="string padding"):
            nn.ConvTranspose1d(4, 2, 3, padding="valid")
        with pytest.raises(ValueError, match="string padding"):
            nn.ConvTranspose3d(4, 2, 3, padding="same")


class TestConvPaddingMode:
    """Tests for the padding_mode keyword across Conv1d/2d/3d."""

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    def test_conv2d_padding_mode_shape(self, mode):
        x = make_tensor((2, 3, 8, 8))
        layer = nn.Conv2d(3, 6, kernel_size=3, padding=1, padding_mode=mode)
        out = layer(x)
        assert out.shape == (2, 6, 8, 8)

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    def test_conv1d_padding_mode_shape(self, mode):
        x = make_tensor((2, 4, 12))
        layer = nn.Conv1d(4, 8, kernel_size=3, padding=1, padding_mode=mode)
        out = layer(x)
        assert out.shape == (2, 8, 12)

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    def test_conv3d_padding_mode_shape(self, mode):
        x = make_tensor((1, 2, 4, 4, 4))
        layer = nn.Conv3d(2, 4, kernel_size=3, padding=1, padding_mode=mode)
        out = layer(x)
        assert out.shape == (1, 4, 4, 4, 4)

    def test_unknown_padding_mode_rejected(self):
        with pytest.raises(ValueError, match="padding_mode"):
            nn.Conv2d(3, 6, kernel_size=3, padding_mode="bogus")
        with pytest.raises(ValueError, match="padding_mode"):
            nn.Conv1d(3, 6, kernel_size=3, padding_mode="symmetric")

    def test_padding_mode_in_repr(self):
        layer = nn.Conv2d(3, 6, kernel_size=3, padding=1, padding_mode="reflect")
        assert "padding_mode='reflect'" in repr(layer)
        # zeros mode is the default — keep extra_repr clean.
        layer_default = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        assert "padding_mode" not in repr(layer_default)


class TestSamePadding:
    def test_same_padding_odd_kernel(self):
        x = make_tensor((2, 3, 8, 8))
        layer = nn.Conv2d(3, 6, kernel_size=3, padding="same")
        out = layer(x)
        assert out.shape == (2, 6, 8, 8)

    def test_same_padding_even_kernel(self):
        # Even kernel: requires asymmetric padding to preserve shape.
        x = make_tensor((2, 3, 7, 7))
        layer = nn.Conv2d(3, 6, kernel_size=4, padding="same")
        out = layer(x)
        assert out.shape == (2, 6, 7, 7)

    def test_same_padding_with_dilation(self):
        x = make_tensor((2, 3, 8, 8))
        layer = nn.Conv2d(3, 6, kernel_size=3, padding="same", dilation=2)
        out = layer(x)
        assert out.shape == (2, 6, 8, 8)

    def test_valid_padding(self):
        x = make_tensor((2, 3, 8, 8))
        layer = nn.Conv2d(3, 6, kernel_size=3, padding="valid")
        out = layer(x)
        assert out.shape == (2, 6, 6, 6)

    def test_same_with_stride_rejected(self):
        with pytest.raises(ValueError, match="stride"):
            nn.Conv2d(3, 6, kernel_size=3, padding="same", stride=2)
        with pytest.raises(ValueError, match="stride"):
            nn.Conv1d(3, 6, kernel_size=3, padding="same", stride=2)

    def test_invalid_string_padding_rejected(self):
        with pytest.raises(ValueError, match="'same' or 'valid'"):
            nn.Conv2d(3, 6, kernel_size=3, padding="full")


class TestGroupedConv:
    def test_depthwise_conv2d_shape(self):
        # Depthwise: groups == in_channels
        Cin = 8
        x = make_tensor((2, Cin, 8, 8))
        layer = nn.Conv2d(Cin, Cin, kernel_size=3, padding=1, groups=Cin)
        out = layer(x)
        assert out.shape == (2, Cin, 8, 8)
        # Weight shape: (Cout, Cin/groups, kH, kW) = (Cin, 1, 3, 3)
        assert layer.weight.shape == (Cin, 1, 3, 3)

    def test_grouped_conv2d_shape(self):
        x = make_tensor((2, 8, 8, 8))
        layer = nn.Conv2d(8, 16, kernel_size=3, padding=1, groups=4)
        out = layer(x)
        assert out.shape == (2, 16, 8, 8)
        # Each group gets Cin/groups = 2 input channels
        assert layer.weight.shape == (16, 2, 3, 3)

    def test_grouped_conv2d_backward(self):
        layer = nn.Conv2d(8, 16, kernel_size=3, padding=1, groups=4)
        x = make_tensor((2, 8, 8, 8), requires_grad=True)
        out = layer(x)
        lucid.sum(out).backward()
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape

    def test_groups_must_divide_channels(self):
        # in_channels must be divisible by groups — caught at first forward.
        layer = nn.Conv2d(8, 16, kernel_size=3, groups=4)
        # Should pass since 8 % 4 == 0
        layer(make_tensor((1, 8, 8, 8)))
