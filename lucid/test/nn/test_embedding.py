"""Tests for Embedding and EmbeddingBag."""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_int_tensor


class TestEmbedding:
    def test_output_shape_1d(self):
        layer = nn.Embedding(num_embeddings=10, embedding_dim=8)
        idx = lucid.tensor([0, 3, 5], dtype=lucid.int32)
        out = layer(idx)
        assert out.shape == (3, 8)

    def test_output_shape_2d(self):
        layer = nn.Embedding(10, 8)
        idx = make_int_tensor((4, 5), low=0, high=10)
        out = layer(idx)
        assert out.shape == (4, 5, 8)

    def test_padding_idx(self):
        layer = nn.Embedding(10, 8, padding_idx=0)
        idx = lucid.tensor([0, 1, 2], dtype=lucid.int32)
        out = layer(idx)
        # padding_idx row should be all zeros
        np.testing.assert_array_almost_equal(out[0].numpy(), np.zeros(8), decimal=5)

    def test_weight_shape(self):
        layer = nn.Embedding(10, 8)
        assert layer.weight.shape == (10, 8)

    def test_backward(self):
        layer = nn.Embedding(10, 8)
        idx = lucid.tensor([1, 2, 3], dtype=lucid.int32)
        out = layer(idx)
        lucid.sum(out).backward()
        # weight gradient should exist
        assert layer.weight.grad is not None

    def test_functional_embedding(self):
        w = lucid.ones(5, 4)
        idx = lucid.tensor([0, 2, 4], dtype=lucid.int32)
        out = F.embedding(idx, w)
        assert out.shape == (3, 4)


class TestEmbeddingBag:
    def test_output_shape_mean_mode(self):
        layer = nn.EmbeddingBag(10, 8, mode="mean")
        idx = make_int_tensor((3, 4), low=0, high=10)
        out = layer(idx)
        assert out.shape == (3, 8)

    def test_output_shape_sum_mode(self):
        layer = nn.EmbeddingBag(10, 8, mode="sum")
        idx = make_int_tensor((3, 4), low=0, high=10)
        out = layer(idx)
        assert out.shape == (3, 8)

    @pytest.mark.parametrize("mode", ["sum", "mean"])
    def test_mode_variants(self, mode):
        layer = nn.EmbeddingBag(10, 8, mode=mode)
        idx = make_int_tensor((2, 5), low=0, high=10)
        out = layer(idx)
        assert out.shape == (2, 8)
