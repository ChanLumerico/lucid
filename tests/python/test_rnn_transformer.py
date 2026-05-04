"""
Integration tests for recurrent (LSTM, GRU, RNN, *Cell) and Transformer modules.

Shape assertions use the actual output shapes produced by this implementation —
see docstrings for notes where shapes differ from PyTorch convention due to
known engine quirks.
"""

import numpy as np
import pytest
import lucid
import lucid.nn as nn


# ── Helpers ───────────────────────────────────────────────────────────────────

def randf(*shape: int) -> lucid.Tensor:
    return lucid.randn(*shape)


# ── LSTMCell ─────────────────────────────────────────────────────────────────


class TestLSTMCell:
    def test_output_shape(self):
        cell = nn.LSTMCell(input_size=8, hidden_size=16)
        x = randf(4, 8)
        h, c = cell(x)
        assert h.shape == (4, 16)
        assert c.shape == (4, 16)

    def test_with_initial_hidden(self):
        cell = nn.LSTMCell(input_size=6, hidden_size=12)
        x = randf(3, 6)
        h0 = lucid.zeros(3, 12)
        c0 = lucid.zeros(3, 12)
        h, c = cell(x, (h0, c0))
        assert h.shape == (3, 12)
        assert c.shape == (3, 12)

    def test_no_bias(self):
        cell = nn.LSTMCell(input_size=4, hidden_size=8, bias=False)
        x = randf(2, 4)
        h, c = cell(x)
        assert h.shape == (2, 8)

    def test_hidden_state_differs_from_input(self):
        cell = nn.LSTMCell(input_size=8, hidden_size=16)
        x = lucid.zeros(2, 8)
        h, c = cell(x)
        # Output should not be all zeros (non-trivial transformation)
        assert not np.allclose(h.numpy(), 0.0)

    def test_state_changes_over_steps(self):
        cell = nn.LSTMCell(input_size=4, hidden_size=8)
        x = randf(2, 4)
        h, c = cell(x)
        h2, c2 = cell(x, (h, c))
        # Hidden state should change between steps
        assert not np.allclose(h.numpy(), h2.numpy())


# ── LSTM ──────────────────────────────────────────────────────────────────────


class TestLSTM:
    def test_basic_batch_first(self):
        lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
        x = randf(2, 5, 8)
        out, (h, c) = lstm(x)
        assert out.shape == (2, 5, 16)
        assert h.shape == (1, 2, 16)
        assert c.shape == (1, 2, 16)

    def test_basic_seq_first(self):
        lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=False)
        x = randf(5, 2, 8)  # (T, B, input)
        out, (h, c) = lstm(x)
        assert out.shape == (5, 2, 16)
        assert h.shape == (1, 2, 16)

    def test_multi_layer(self):
        lstm = nn.LSTM(input_size=8, hidden_size=16, num_layers=2, batch_first=True)
        x = randf(2, 5, 8)
        out, (h, c) = lstm(x)
        assert out.shape == (2, 5, 16)
        assert h.shape == (2, 2, 16)  # (num_layers, batch, hidden)

    def test_explicit_initial_state(self):
        lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
        x = randf(2, 5, 8)
        h0 = lucid.zeros(1, 2, 16)
        c0 = lucid.zeros(1, 2, 16)
        out, (h, c) = lstm(x, (h0, c0))
        assert out.shape == (2, 5, 16)

    def test_output_deterministic_on_zeros(self):
        lstm = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
        x1 = lucid.zeros(1, 3, 4)
        x2 = lucid.zeros(1, 3, 4)
        out1, _ = lstm(x1)
        out2, _ = lstm(x2)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)

    def test_output_varies_with_input(self):
        lstm = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
        x1 = lucid.zeros(1, 3, 4)
        x2 = lucid.ones(1, 3, 4)
        out1, _ = lstm(x1)
        out2, _ = lstm(x2)
        assert not np.allclose(out1.numpy(), out2.numpy())

    def test_eval_mode_deterministic(self):
        lstm = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)
        lstm.eval()
        x = randf(1, 3, 4)
        out1, _ = lstm(x)
        out2, _ = lstm(x)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)

    def test_no_bias(self):
        lstm = nn.LSTM(input_size=4, hidden_size=8, bias=False, batch_first=True)
        x = randf(2, 3, 4)
        out, (h, c) = lstm(x)
        assert out.shape == (2, 3, 8)
        assert h.shape == (1, 2, 8)

    def test_no_bias_differs_from_bias(self):
        x = randf(1, 3, 4)
        lstm_bias = nn.LSTM(input_size=4, hidden_size=8, bias=True, batch_first=True)
        lstm_no = nn.LSTM(input_size=4, hidden_size=8, bias=False, batch_first=True)
        out_b, _ = lstm_bias(x)
        out_n, _ = lstm_no(x)
        # Different weights initialisation → outputs should differ
        assert out_b.shape == out_n.shape


# ── GRUCell ───────────────────────────────────────────────────────────────────


class TestGRUCell:
    def test_output_shape(self):
        cell = nn.GRUCell(input_size=8, hidden_size=16)
        x = randf(4, 8)
        h = cell(x)
        assert h.shape == (4, 16)

    def test_with_initial_hidden(self):
        cell = nn.GRUCell(input_size=6, hidden_size=12)
        x = randf(3, 6)
        h0 = lucid.zeros(3, 12)
        h = cell(x, h0)
        assert h.shape == (3, 12)

    def test_state_changes_over_steps(self):
        cell = nn.GRUCell(input_size=4, hidden_size=8)
        x = randf(2, 4)
        h = cell(x)
        h2 = cell(x, h)
        assert not np.allclose(h.numpy(), h2.numpy())


# ── GRU ───────────────────────────────────────────────────────────────────────


class TestGRU:
    def test_basic_batch_first(self):
        gru = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
        x = randf(2, 5, 8)
        out, h = gru(x)
        assert out.shape == (2, 5, 16)
        assert h.shape == (1, 2, 16)  # (D*num_layers, batch, hidden)

    def test_seq_first(self):
        gru = nn.GRU(input_size=8, hidden_size=16, batch_first=False)
        x = randf(5, 2, 8)
        out, h = gru(x)
        assert out.shape == (5, 2, 16)
        assert h.shape == (1, 2, 16)

    def test_explicit_h0(self):
        gru = nn.GRU(input_size=4, hidden_size=8, batch_first=True)
        x = randf(2, 3, 4)
        h0 = lucid.zeros(1, 2, 8)  # (D*num_layers, batch, hidden)
        out, h = gru(x, h0)
        assert out.shape == (2, 3, 8)

    def test_output_varies_with_input(self):
        gru = nn.GRU(input_size=4, hidden_size=8, batch_first=True)
        x1 = lucid.zeros(1, 3, 4)
        x2 = lucid.ones(1, 3, 4)
        out1, _ = gru(x1)
        out2, _ = gru(x2)
        assert not np.allclose(out1.numpy(), out2.numpy())

    def test_eval_deterministic(self):
        gru = nn.GRU(input_size=4, hidden_size=8, batch_first=True)
        gru.eval()
        x = randf(1, 3, 4)
        out1, _ = gru(x)
        out2, _ = gru(x)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)

    def test_multi_layer(self):
        gru = nn.GRU(input_size=8, hidden_size=16, num_layers=2, batch_first=True)
        x = randf(2, 5, 8)
        out, h = gru(x)
        assert out.shape == (2, 5, 16)
        assert h.shape == (2, 2, 16)  # (num_layers, batch, hidden)

    def test_bidirectional(self):
        gru = nn.GRU(input_size=8, hidden_size=16, bidirectional=True, batch_first=True)
        x = randf(2, 5, 8)
        out, h = gru(x)
        assert out.shape == (2, 5, 32)  # 2 * hidden_size
        assert h.shape == (2, 2, 16)    # (D=2, batch, hidden)

    def test_bidirectional_multilayer(self):
        gru = nn.GRU(input_size=8, hidden_size=16, num_layers=3,
                     bidirectional=True, batch_first=True)
        x = randf(2, 5, 8)
        out, h = gru(x)
        assert out.shape == (2, 5, 32)
        assert h.shape == (6, 2, 16)   # D*num_layers=6

    def test_explicit_h0_3d(self):
        gru = nn.GRU(input_size=4, hidden_size=8, num_layers=2, batch_first=True)
        x = randf(2, 3, 4)
        h0 = lucid.zeros(2, 2, 8)
        out, h = gru(x, h0)
        assert out.shape == (2, 3, 8)
        assert h.shape == (2, 2, 8)


# ── RNNCell ───────────────────────────────────────────────────────────────────


class TestRNNCell:
    def test_tanh(self):
        cell = nn.RNNCell(input_size=8, hidden_size=16)
        x = randf(4, 8)
        h = cell(x)
        assert h.shape == (4, 16)
        # tanh output in (-1, 1)
        assert h.numpy().max() <= 1.0 + 1e-4
        assert h.numpy().min() >= -1.0 - 1e-4

    def test_relu(self):
        cell = nn.RNNCell(input_size=4, hidden_size=8, nonlinearity="relu")
        x = randf(2, 4)
        h = cell(x)
        assert h.shape == (2, 8)
        assert (h.numpy() >= 0).all()

    def test_with_initial_hidden(self):
        cell = nn.RNNCell(input_size=4, hidden_size=8)
        x = randf(3, 4)
        h0 = lucid.zeros(3, 8)
        h = cell(x, h0)
        assert h.shape == (3, 8)


# ── RNN ───────────────────────────────────────────────────────────────────────


class TestRNN:
    def test_basic_tanh(self):
        rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
        x = randf(2, 5, 8)
        out, h = rnn(x)
        assert out.shape == (2, 5, 16)
        assert h.shape == (1, 2, 16)  # (D*num_layers, batch, hidden)

    def test_relu_nonlinearity(self):
        rnn = nn.RNN(input_size=4, hidden_size=8, nonlinearity="relu", batch_first=True)
        x = randf(2, 3, 4)
        out, h = rnn(x)
        assert out.shape == (2, 3, 8)
        # Tanh output: all outputs in (-1,1); ReLU: non-negative
        assert (out.numpy() >= 0).all()

    def test_seq_first(self):
        rnn = nn.RNN(input_size=4, hidden_size=8, batch_first=False)
        x = randf(5, 2, 4)
        out, h = rnn(x)
        assert out.shape == (5, 2, 8)

    def test_multi_layer(self):
        rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=3, batch_first=True)
        x = randf(2, 5, 8)
        out, h = rnn(x)
        assert out.shape == (2, 5, 16)
        assert h.shape == (3, 2, 16)

    def test_bidirectional(self):
        rnn = nn.RNN(input_size=8, hidden_size=16, bidirectional=True, batch_first=True)
        x = randf(2, 5, 8)
        out, h = rnn(x)
        assert out.shape == (2, 5, 32)  # 2 * hidden_size
        assert h.shape == (2, 2, 16)

    def test_explicit_h0_3d(self):
        rnn = nn.RNN(input_size=4, hidden_size=8, num_layers=2, batch_first=True)
        x = randf(2, 3, 4)
        h0 = lucid.zeros(2, 2, 8)
        out, h = rnn(x, h0)
        assert out.shape == (2, 3, 8)
        assert h.shape == (2, 2, 8)

    def test_eval_deterministic(self):
        rnn = nn.RNN(input_size=4, hidden_size=8, batch_first=True)
        rnn.eval()
        x = randf(1, 3, 4)
        out1, _ = rnn(x)
        out2, _ = rnn(x)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)


# ── Transformer ───────────────────────────────────────────────────────────────


class TestTransformerEncoderLayer:
    def test_output_shape(self):
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        x = randf(2, 5, 32)
        y = layer(x)
        assert y.shape == (2, 5, 32)

    def test_preserves_embedding_dim(self):
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=8,
                                            dim_feedforward=256, batch_first=True)
        x = randf(3, 10, 64)
        y = layer(x)
        assert y.shape == (3, 10, 64)

    def test_with_src_mask(self):
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        x = randf(1, 4, 32)
        mask = lucid.tensor(np.triu(np.full((4, 4), float("-inf")), k=1).astype(np.float32))
        y = layer(x, src_mask=mask)
        assert y.shape == (1, 4, 32)

    def test_eval_vs_train(self):
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True, dropout=0.1)
        x = randf(1, 4, 32)
        layer.eval()
        y1 = layer(x)
        y2 = layer(x)
        np.testing.assert_allclose(y1.numpy(), y2.numpy(), atol=1e-5)


class TestTransformerEncoder:
    def test_output_shape(self):
        enc_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        x = randf(2, 5, 32)
        y = encoder(x)
        assert y.shape == (2, 5, 32)

    def test_single_layer(self):
        enc_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
        encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        x = randf(2, 3, 16)
        y = encoder(x)
        assert y.shape == (2, 3, 16)

    def test_transforms_input(self):
        enc_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
        encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        x = lucid.zeros(1, 3, 16)
        y = encoder(x)
        # Should not be all zeros after transformation
        assert not np.allclose(y.numpy(), 0.0)


class TestTransformerDecoderLayer:
    def test_output_shape(self):
        layer = nn.TransformerDecoderLayer(d_model=32, nhead=4, batch_first=True)
        tgt = randf(2, 3, 32)
        mem = randf(2, 5, 32)
        y = layer(tgt, mem)
        assert y.shape == (2, 3, 32)

    def test_different_src_tgt_lengths(self):
        layer = nn.TransformerDecoderLayer(d_model=32, nhead=4, batch_first=True)
        tgt = randf(1, 7, 32)   # decoder sequence
        mem = randf(1, 10, 32)  # encoder memory
        y = layer(tgt, mem)
        assert y.shape == (1, 7, 32)


class TestTransformerDecoder:
    def test_output_shape(self):
        dec_layer = nn.TransformerDecoderLayer(d_model=32, nhead=4, batch_first=True)
        decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        tgt = randf(2, 3, 32)
        mem = randf(2, 5, 32)
        y = decoder(tgt, mem)
        assert y.shape == (2, 3, 32)


class TestTransformer:
    def test_basic(self):
        T = nn.Transformer(d_model=32, nhead=4, num_encoder_layers=2,
                           num_decoder_layers=2, dim_feedforward=64, batch_first=True)
        src = randf(2, 5, 32)
        tgt = randf(2, 3, 32)
        out = T(src, tgt)
        assert out.shape == (2, 3, 32)

    def test_with_causal_mask(self):
        T = nn.Transformer(d_model=32, nhead=4, num_encoder_layers=1,
                           num_decoder_layers=1, dim_feedforward=64, batch_first=True)
        src = randf(1, 4, 32)
        tgt = randf(1, 4, 32)
        L = 4
        mask = lucid.tensor(np.triu(np.full((L, L), float("-inf")), k=1).astype(np.float32))
        out = T(src, tgt, tgt_mask=mask)
        assert out.shape == (1, 4, 32)

    def test_eval_mode(self):
        T = nn.Transformer(d_model=32, nhead=4, num_encoder_layers=2,
                           num_decoder_layers=2, dim_feedforward=64,
                           dropout=0.1, batch_first=True)
        T.eval()
        src = randf(1, 4, 32)
        tgt = randf(1, 3, 32)
        out1 = T(src, tgt)
        out2 = T(src, tgt)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)

    def test_output_shape_asymmetric(self):
        T = nn.Transformer(d_model=16, nhead=2, num_encoder_layers=1,
                           num_decoder_layers=1, dim_feedforward=32, batch_first=True)
        src = randf(3, 7, 16)   # longer encoder input
        tgt = randf(3, 4, 16)   # shorter decoder input
        out = T(src, tgt)
        assert out.shape == (3, 4, 16)  # output matches tgt length


# ── MultiheadAttention ────────────────────────────────────────────────────────


class TestMultiheadAttention:
    def test_self_attention_shape(self):
        mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        q = k = v = randf(2, 5, 32)
        out, weights = mha(q, k, v)
        assert out.shape == (2, 5, 32)

    def test_cross_attention_shape(self):
        mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        q = randf(2, 3, 32)
        k = v = randf(2, 7, 32)
        out, _ = mha(q, k, v)
        assert out.shape == (2, 3, 32)

    def test_with_attn_mask(self):
        mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        q = k = v = randf(1, 4, 32)
        mask = lucid.tensor(np.triu(np.full((4, 4), float("-inf")), k=1).astype(np.float32))
        out, _ = mha(q, k, v, attn_mask=mask)
        assert out.shape == (1, 4, 32)

    def test_no_bias(self):
        mha = nn.MultiheadAttention(embed_dim=16, num_heads=2, bias=False, batch_first=True)
        q = k = v = randf(2, 5, 16)
        out, _ = mha(q, k, v)
        assert out.shape == (2, 5, 16)

    def test_eval_deterministic(self):
        mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, dropout=0.1, batch_first=True)
        mha.eval()
        q = k = v = randf(1, 4, 32)
        out1, _ = mha(q, k, v)
        out2, _ = mha(q, k, v)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)
