"""Embedding / RNN / containers / shape modules / padding / transformer."""

import numpy as np

import lucid
import lucid.nn as nn
from lucid.test._helpers.compare import assert_close


class TestEmbedding:
    def test_shape(self) -> None:
        m = nn.Embedding(num_embeddings=10, embedding_dim=4)
        idx = lucid.tensor([0, 1, 2, 3], dtype=lucid.int64)
        assert m(idx).shape == (4, 4)

    def test_padding_idx_row_zero(self) -> None:
        m = nn.Embedding(10, 4, padding_idx=0)
        # ``init`` zeroes the pad row.
        np.testing.assert_array_equal(m.weight[0].numpy(), [0.0, 0.0, 0.0, 0.0])


class TestSequential:
    def test_chain(self) -> None:
        net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        out = net(lucid.zeros(3, 4))
        assert out.shape == (3, 2)


class TestModuleList:
    def test_basic(self) -> None:
        ml = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        assert len(ml) == 3


class TestFlatten:
    def test_basic(self) -> None:
        m = nn.Flatten()
        out = m(lucid.zeros(2, 3, 4, 5))
        # Default flattens dim 1+ → (2, 60).
        assert out.shape == (2, 60)


class TestRNNFamily:
    def test_lstm_shape(self) -> None:
        m = nn.LSTM(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
        x = lucid.zeros(2, 5, 4)
        out, _ = m(x)
        assert out.shape == (2, 5, 8)

    def test_gru_shape(self) -> None:
        m = nn.GRU(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
        x = lucid.zeros(2, 5, 4)
        out, _ = m(x)
        assert out.shape == (2, 5, 8)

    def test_rnn_shape(self) -> None:
        m = nn.RNN(
            input_size=4,
            hidden_size=8,
            num_layers=1,
            batch_first=True,
            nonlinearity="tanh",
        )
        x = lucid.zeros(2, 5, 4)
        out, _ = m(x)
        assert out.shape == (2, 5, 8)


class TestMultiheadAttention:
    def test_shape(self) -> None:
        m = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        q = lucid.zeros(1, 4, 8)
        out, _ = m(q, q, q)
        assert out.shape == (1, 4, 8)


class TestGroupedQueryAttention:
    """Grouped-query / multi-query attention (``num_kv_heads`` on MHA)."""

    def test_repeat_kv_grouping(self) -> None:
        # repeat_kv(x, n): result head h must equal source head h // n.
        x = lucid.randn(2, 3, 5, 4)  # (B, num_kv_heads=3, T, D)
        r = nn.functional.repeat_kv(x, 2)
        assert tuple(r.shape) == (2, 6, 5, 4)
        for h in range(6):
            diff = (r[:, h] - x[:, h // 2]).abs().sum().item()
            assert float(diff) == 0.0
        # n_rep == 1 is a no-op (same object).
        assert nn.functional.repeat_kv(x, 1) is x

    def test_gqa_shapes_and_smaller_kv_projection(self) -> None:
        E, H = 64, 8
        gqa = nn.MultiheadAttention(E, H, num_kv_heads=2, batch_first=True)
        assert gqa.num_kv_heads == 2
        # K/V project to num_kv_heads*head_dim = 2*8 = 16 (vs embed_dim 64).
        assert tuple(gqa.k_proj_weight.shape) == (16, 64)
        assert gqa.in_proj_weight is None  # GQA forces separate projections
        x = lucid.randn(2, 5, E)
        out, w = gqa(x, x, x, need_weights=True)
        assert out.shape == (2, 5, E)
        assert w.shape == (2, 5, 5)

    def test_mqa_is_num_kv_heads_one(self) -> None:
        mqa = nn.MultiheadAttention(32, 4, num_kv_heads=1, batch_first=True)
        assert mqa.num_kv_heads == 1
        assert tuple(mqa.k_proj_weight.shape) == (8, 32)  # 1 head * head_dim 8
        x = lucid.randn(1, 3, 32)
        out, _ = mqa(x, x, x, need_weights=False)
        assert out.shape == (1, 3, 32)

    def test_cache_stores_num_kv_heads(self) -> None:
        # The GQA win: the K/V cache holds the smaller num_kv_heads set.
        from lucid.utils.cache import DynamicCache

        gqa = nn.MultiheadAttention(64, 8, num_kv_heads=2, batch_first=True)
        cache = DynamicCache()
        x = lucid.randn(2, 1, 64)
        gqa(
            x,
            x,
            x,
            use_cache=True,
            past_key_value=cache,
            layer_idx=0,
            need_weights=False,
        )
        # (B, num_kv_heads=2, T, head_dim=8) — NOT (B, 8, ...)
        assert tuple(cache.key_cache[0].shape) == (2, 2, 1, 8)

    def test_num_kv_equals_num_heads_is_standard_mha(self) -> None:
        lucid.manual_seed(0)
        a = nn.MultiheadAttention(32, 4, batch_first=True)
        lucid.manual_seed(0)
        b = nn.MultiheadAttention(32, 4, num_kv_heads=4, batch_first=True)
        assert b.in_proj_weight is not None  # uses the combined path
        x = lucid.randn(2, 4, 32)
        oa, _ = a(x, x, x, need_weights=False)
        ob, _ = b(x, x, x, need_weights=False)
        assert float((oa - ob).abs().max().item()) == 0.0

    def test_invalid_num_kv_heads_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="divisible by num_kv_heads"):
            nn.MultiheadAttention(64, 8, num_kv_heads=3)  # 8 % 3 != 0

    def test_gqa_matches_hand_rolled(self) -> None:
        # Strongest correctness check: the module's GQA wiring must equal a
        # from-scratch grouped-query attention (project → repeat_kv → SDPA → out)
        # built from the SAME weights with lucid's own ops.
        import lucid.nn.functional as F

        E, H, KV = 32, 4, 2
        lucid.manual_seed(1)
        gqa = nn.MultiheadAttention(E, H, num_kv_heads=KV, bias=False, batch_first=True)
        x = lucid.randn(2, 6, E)
        out, _ = gqa(x, x, x, need_weights=False)

        # Hand-rolled reference from the module's own projection weights.
        hd, nrep = E // H, H // KV
        q = F.linear(x, gqa.q_proj_weight)  # (B,T,E)
        k = F.linear(x, gqa.k_proj_weight)  # (B,T,KV*hd)
        v = F.linear(x, gqa.v_proj_weight)
        B, T = x.shape[0], x.shape[1]
        qh = q.reshape(B, T, H, hd).permute([0, 2, 1, 3])
        kh = k.reshape(B, T, KV, hd).permute([0, 2, 1, 3])
        vh = v.reshape(B, T, KV, hd).permute([0, 2, 1, 3])
        kh = F.repeat_kv(kh, nrep)
        vh = F.repeat_kv(vh, nrep)
        attn = F.scaled_dot_product_attention(qh, kh, vh)
        merged = attn.permute([0, 2, 1, 3]).reshape(B, T, E)
        ref = F.linear(merged, gqa.out_proj_weight, gqa.out_proj_bias)

        assert_close(out, ref, atol=1e-5)

    def test_grouped_query_attention_alias(self) -> None:
        # nn.GroupedQueryAttention(E, H, KV) == MultiheadAttention(num_kv_heads=KV).
        lucid.manual_seed(3)
        a = nn.GroupedQueryAttention(64, 8, 2, batch_first=True)
        lucid.manual_seed(3)
        b = nn.MultiheadAttention(64, 8, num_kv_heads=2, batch_first=True)
        assert a.num_kv_heads == 2
        assert isinstance(a, nn.MultiheadAttention)
        x = lucid.randn(2, 5, 64)
        oa, _ = a(x, x, x, need_weights=False)
        ob, _ = b(x, x, x, need_weights=False)
        assert float((oa - ob).abs().max().item()) == 0.0

    def test_multi_query_attention_alias(self) -> None:
        # nn.MultiQueryAttention(E, H) == MultiheadAttention(num_kv_heads=1).
        lucid.manual_seed(4)
        a = nn.MultiQueryAttention(32, 4, batch_first=True)
        lucid.manual_seed(4)
        b = nn.MultiheadAttention(32, 4, num_kv_heads=1, batch_first=True)
        assert a.num_kv_heads == 1
        x = lucid.randn(1, 3, 32)
        oa, _ = a(x, x, x, need_weights=False)
        ob, _ = b(x, x, x, need_weights=False)
        assert float((oa - ob).abs().max().item()) == 0.0


class TestPadding:
    def test_circular_pad_1d(self) -> None:
        m = nn.CircularPad1d(1)
        x = lucid.tensor([[[1.0, 2.0, 3.0]]])
        out = m(x).numpy()
        np.testing.assert_array_equal(out, [[[3.0, 1.0, 2.0, 3.0, 1.0]]])

    def test_reflection_pad_3d(self) -> None:
        m = nn.ReflectionPad3d(1)
        x = lucid.zeros(1, 1, 3, 3, 3)
        assert m(x).shape == (1, 1, 5, 5, 5)


class TestUpsampling:
    def test_pixel_shuffle(self) -> None:
        m = nn.PixelShuffle(upscale_factor=2)
        out = m(lucid.zeros(1, 4, 4, 4))
        assert out.shape == (1, 1, 8, 8)

    def test_channel_shuffle(self) -> None:
        m = nn.ChannelShuffle(groups=2)
        x = lucid.tensor([[[10.0], [11.0], [12.0], [13.0]]])
        out = m(x).numpy()
        np.testing.assert_array_equal(out, [[[10.0], [12.0], [11.0], [13.0]]])


class TestTransformerBlock:
    def test_encoder_layer_shape(self) -> None:
        m = nn.TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True)
        out = m(lucid.zeros(1, 4, 8))
        assert out.shape == (1, 4, 8)
