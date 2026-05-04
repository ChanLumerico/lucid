"""Tests for Linear, Bilinear, and F.linear."""

import pytest
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


class TestFLinear:
    def test_output_shape(self):
        x = make_tensor((4, 8))
        w = make_tensor((6, 8))
        b = make_tensor((6,))
        out = F.linear(x, w, b)
        assert out.shape == (4, 6)

    def test_no_bias(self):
        x = make_tensor((3, 5))
        w = make_tensor((4, 5))
        out = F.linear(x, w)
        assert out.shape == (3, 4)

    def test_batch_linear(self):
        x = make_tensor((2, 4, 8))
        w = make_tensor((6, 8))
        out = F.linear(x, w)
        assert out.shape == (2, 4, 6)

    def test_identity_weight(self):
        x = make_tensor((3, 4))
        w = lucid.eye(4)
        out = F.linear(x, w)
        assert_close(out, x)


class TestNNLinear:
    def test_construction(self):
        layer = nn.Linear(8, 4)
        assert layer.weight.shape == (4, 8)
        assert layer.bias.shape == (4,)

    def test_forward_output_shape(self):
        layer = nn.Linear(8, 4)
        x = make_tensor((3, 8))
        out = layer(x)
        assert out.shape == (3, 4)

    def test_no_bias_option(self):
        layer = nn.Linear(8, 4, bias=False)
        assert layer.bias is None

    def test_parameters_count(self):
        layer = nn.Linear(8, 4)
        params = list(layer.parameters())
        assert len(params) == 2  # weight + bias

    def test_parameters_count_no_bias(self):
        layer = nn.Linear(8, 4, bias=False)
        params = list(layer.parameters())
        assert len(params) == 1

    def test_backward_computes_grad(self):
        layer = nn.Linear(4, 2)
        x = make_tensor((3, 4), requires_grad=True)
        out = layer(x)
        loss = lucid.sum(out)
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (3, 4)

    def test_train_eval_mode(self):
        layer = nn.Linear(4, 2)
        layer.eval()
        assert not layer.training
        layer.train()
        assert layer.training


class TestNNBilinear:
    def test_output_shape(self):
        layer = nn.Bilinear(4, 5, 3)
        x1 = make_tensor((2, 4))
        x2 = make_tensor((2, 5))
        out = layer(x1, x2)
        assert out.shape == (2, 3)
