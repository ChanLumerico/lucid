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


class TestEmbeddingContract:
    def test_padding_idx_zero_init(self):
        m = nn.Embedding(8, 4, padding_idx=2)
        assert (m.weight.numpy()[2] == 0.0).all()

    def test_padding_idx_zero_grad(self):
        # Backward over indices that include the pad row should leave that row's
        # gradient at zero.
        import numpy as np
        from lucid._C import engine as _C_engine
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        m = nn.Embedding(5, 4, padding_idx=2)
        rng = np.random.default_rng(0)
        W = rng.standard_normal((5, 4)).astype(np.float32)
        m.weight._impl = _iwg(
            _C_engine.TensorImpl(W, _C_engine.Device.CPU, False), True
        )
        x = lucid.tensor([0, 2, 1, 2, 0], dtype=lucid.int32)
        out = m(x)
        out.sum().backward()
        assert (m.weight.grad.numpy()[2] == 0.0).all()
        # Other rows must have gradients.
        assert m.weight.grad.numpy()[0].any()

    def test_padding_idx_normalises_negative(self):
        m = nn.Embedding(5, 4, padding_idx=-1)
        assert m.padding_idx == 4

    def test_padding_idx_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="padding_idx"):
            nn.Embedding(5, 4, padding_idx=10)

    def test_max_norm_renormalises_rows(self):
        import numpy as np

        m = nn.Embedding(5, 4, max_norm=1.0)
        # Force a large weight row.
        m.weight._impl.data_as_python()[:] = np.array(
            [[10.0, 0.0, 0.0, 0.0]] + [[0.0, 0.0, 0.0, 0.0]] * 4,
            dtype=np.float32,
        )
        m(lucid.tensor([0, 1], dtype=lucid.int32))
        assert abs(np.linalg.norm(m.weight.numpy()[0]) - 1.0) < 1e-5

    def test_scale_grad_by_freq_rejected(self):
        with pytest.raises(NotImplementedError, match="scale_grad_by_freq"):
            nn.Embedding(5, 4, scale_grad_by_freq=True)
