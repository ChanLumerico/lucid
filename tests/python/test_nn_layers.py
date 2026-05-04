"""
Tests for nn layer modules: activations, conv, normalization, pooling,
padding, upsampling, attention, RNN cells, Transformer.
"""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
from conftest import assert_close


class TestActivationModules:
    def test_relu(self):
        m = nn.ReLU()
        x = lucid.tensor([-1.0, 0.0, 1.0])
        out = m(x)
        assert_close(out.numpy(), np.array([0.0, 0.0, 1.0]))

    def test_prelu(self):
        m = nn.PReLU(1)
        x = lucid.randn(3, 4)
        assert m(x).shape == x.shape

    def test_threshold(self):
        m = nn.Threshold(0.5, -1.0)
        x = lucid.tensor([0.0, 0.6, 1.0])
        out = m(x)
        assert out.shape == (3,)

    def test_hardtanh(self):
        m = nn.Hardtanh(-1.0, 1.0)
        x = lucid.tensor([-2.0, 0.0, 2.0])
        out = m(x)
        assert_close(out.numpy(), np.array([-1.0, 0.0, 1.0]))

    def test_logsigmoid(self):
        m = nn.LogSigmoid()
        x = lucid.randn(3, 4)
        out = m(x)
        assert out.shape == x.shape
        assert np.all(out.numpy() <= 0)  # log(sigmoid(x)) <= 0 always

    def test_softsign(self):
        m = nn.Softsign()
        x = lucid.tensor([0.0, 1.0, -1.0])
        out = m(x)
        assert_close(out.numpy(), np.array([0.0, 0.5, -0.5]))

    def test_softmin(self):
        m = nn.Softmin(dim=1)
        x = lucid.randn(3, 5)
        out = m(x)
        assert_close(out.numpy().sum(axis=1), np.ones(3))

    def test_glu(self):
        m = nn.GLU(dim=1)
        x = lucid.randn(2, 8)
        assert m(x).shape == (2, 4)


class TestPaddingModules:
    def test_zero_pad2d_int(self):
        m = nn.ZeroPad2d(2)
        x = lucid.randn(1, 3, 4, 4)
        assert m(x).shape == (1, 3, 8, 8)

    def test_zero_pad2d_tuple(self):
        m = nn.ZeroPad2d((1, 2, 3, 4))
        x = lucid.randn(1, 3, 4, 4)
        out = m(x)
        # W: 4+1+2=7, H: 4+3+4=11
        assert out.shape == (1, 3, 11, 7)

    def test_constant_pad2d(self):
        m = nn.ConstantPad2d(1, 999.0)
        x = lucid.zeros(1, 1, 2, 2)
        out = m(x)
        assert out.shape == (1, 1, 4, 4)
        # Corners should be 999
        assert abs(out.numpy()[0, 0, 0, 0] - 999.0) < 1e-5

    def test_replication_pad2d(self):
        m = nn.ReplicationPad2d(1)
        x = lucid.randn(1, 3, 4, 4)
        out = m(x)
        assert out.shape == (1, 3, 6, 6)


class TestUpsamplingModules:
    def test_upsample_nearest(self):
        m = nn.Upsample(size=(8, 8), mode="nearest")
        x = lucid.randn(1, 3, 4, 4)
        assert m(x).shape == (1, 3, 8, 8)

    def test_pixel_shuffle(self):
        r = 2
        m = nn.PixelShuffle(r)
        x = lucid.randn(1, 4, 3, 3)  # 4 = 1*r^2
        out = m(x)
        assert out.shape == (1, 1, 6, 6)

    def test_pixel_unshuffle(self):
        r = 2
        m = nn.PixelUnshuffle(r)
        x = lucid.randn(1, 1, 6, 6)
        out = m(x)
        assert out.shape == (1, 4, 3, 3)

    def test_pixel_shuffle_unshuffle_roundtrip(self):
        x = lucid.randn(2, 4, 4, 4)
        ps = nn.PixelShuffle(2)
        pus = nn.PixelUnshuffle(2)
        out = pus(ps(x))
        assert out.shape == x.shape


class TestRNNCells:
    def test_lstm_cell(self):
        cell = nn.LSTMCell(4, 8)
        x = lucid.randn(3, 4)
        h, c = cell(x)
        assert h.shape == (3, 8)
        assert c.shape == (3, 8)

    def test_lstm_cell_with_hidden(self):
        cell = nn.LSTMCell(4, 8)
        x = lucid.randn(3, 4)
        h0 = lucid.zeros(3, 8)
        c0 = lucid.zeros(3, 8)
        h, c = cell(x, (h0, c0))
        assert h.shape == (3, 8)

    def test_gru_cell(self):
        cell = nn.GRUCell(4, 8)
        x = lucid.randn(3, 4)
        h = cell(x)
        assert h.shape == (3, 8)

    def test_rnn_cell_tanh(self):
        cell = nn.RNNCell(4, 8, nonlinearity="tanh")
        x = lucid.randn(3, 4)
        h = cell(x)
        assert h.shape == (3, 8)

    def test_gru_full(self):
        gru = nn.GRU(4, 8, batch_first=True)
        x = lucid.randn(2, 5, 4)
        out, h = gru(x)
        assert out.shape == (2, 5, 8)
        assert h.shape == (1, 2, 8)  # (D*num_layers, B, H)

    def test_rnn_full(self):
        rnn = nn.RNN(4, 8, batch_first=True)
        x = lucid.randn(2, 5, 4)
        out, h = rnn(x)
        assert out.shape == (2, 5, 8)


class TestTransformer:
    def test_encoder_layer(self):
        layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32)
        src = lucid.randn(5, 3, 16)  # (T, B, d)
        out = layer(src)
        assert out.shape == (5, 3, 16)

    def test_encoder(self):
        enc_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32)
        enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        src = lucid.randn(5, 3, 16)
        out = enc(src)
        assert out.shape == (5, 3, 16)

    def test_decoder(self):
        enc_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32)
        dec_layer = nn.TransformerDecoderLayer(d_model=16, nhead=2, dim_feedforward=32)
        enc = nn.TransformerEncoder(enc_layer, num_layers=1)
        dec = nn.TransformerDecoder(dec_layer, num_layers=1)
        src = lucid.randn(5, 3, 16)
        tgt = lucid.randn(4, 3, 16)
        memory = enc(src)
        out = dec(tgt, memory)
        assert out.shape == (4, 3, 16)

    def test_full_transformer(self):
        t = nn.Transformer(d_model=16, nhead=2, num_encoder_layers=2,
                           num_decoder_layers=2, dim_feedforward=32)
        src = lucid.randn(5, 3, 16)
        tgt = lucid.randn(4, 3, 16)
        out = t(src, tgt)
        assert out.shape == (4, 3, 16)

    def test_multihead_attention_cross(self):
        mha = nn.MultiheadAttention(16, 2)
        q = lucid.randn(4, 3, 16)
        kv = lucid.randn(6, 3, 16)
        out, _ = mha(q, kv, kv)
        assert out.shape == (4, 3, 16)


class TestTensorConvenience:
    def test_fill_(self):
        x = lucid.randn(3, 4)
        x.fill_(7.0)
        assert np.all(x.numpy() == 7.0)

    def test_copy_(self):
        x = lucid.zeros(3)
        y = lucid.ones(3)
        x.copy_(y)
        assert np.all(x.numpy() == 1.0)

    def test_flip_1d(self):
        x = lucid.tensor([1.0, 2.0, 3.0])
        out = x.flip(0)
        assert_close(out.numpy(), np.array([3.0, 2.0, 1.0]))

    def test_flip_2d(self):
        x = lucid.randn(3, 4)
        ref = x.numpy().copy()
        assert_close(x.flip(0).numpy(), ref[::-1])

    def test_index_select(self):
        x = lucid.randn(5, 4)
        idx = lucid.Tensor(np.array([0, 2, 4], dtype=np.int32))
        sel = x.index_select(0, idx)
        assert sel.shape == (3, 4)
        assert_close(sel.numpy()[0], x.numpy()[0])

    def test_masked_select(self):
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        mask = lucid.Tensor(np.array([True, False, True, False]))
        sel = x.masked_select(mask)
        assert sel.shape == (2,)
        assert_close(sel.numpy(), np.array([1.0, 3.0]))

    def test_lerp(self):
        a = lucid.zeros(3)
        b = lucid.ones(3)
        out = a.lerp(b, 0.5)
        assert_close(out.numpy(), np.full(3, 0.5))

    def test_bmm(self):
        a = lucid.randn(2, 3, 4)
        b = lucid.randn(2, 4, 5)
        out = a.bmm(b)
        assert out.shape == (2, 3, 5)

    def test_addmm(self):
        bias = lucid.zeros(2, 3)
        m1 = lucid.ones(2, 4)
        m2 = lucid.ones(4, 3)
        out = bias.addmm(m1, m2)  # 0 + 1*ones@ones = 4*ones
        assert out.shape == (2, 3)
        assert_close(out.numpy(), np.full((2, 3), 4.0))

    def test_diff(self):
        x = lucid.tensor([1.0, 4.0, 9.0, 16.0])
        d = x.diff(n=1, dim=0)
        assert_close(d.numpy(), np.array([3.0, 5.0, 7.0]))

    def test_expand_as(self):
        a = lucid.ones(1, 4)
        b = lucid.zeros(3, 4)
        out = a.expand_as(b)
        assert out.shape == (3, 4)

    def test_view_as(self):
        a = lucid.randn(2, 6)
        b = lucid.zeros(3, 4)
        out = a.view_as(b)
        assert out.shape == (3, 4)

    def test_type_as(self):
        a = lucid.randn(3)
        b = lucid.zeros(2)
        out = a.type_as(b)
        assert out.dtype == b.dtype


class TestFunctionalNew:
    def test_normalize(self):
        import lucid.nn.functional as F
        x = lucid.randn(3, 4)
        n = F.normalize(x, p=2.0, dim=1)
        norms = np.linalg.norm(n.numpy(), axis=1)
        assert_close(norms, np.ones(3))

    def test_cosine_similarity(self):
        import lucid.nn.functional as F
        a = lucid.randn(3, 4)
        b = lucid.randn(3, 4)
        cs = F.cosine_similarity(a, b, dim=1)
        assert cs.shape == (3,)
        assert np.all(np.abs(cs.numpy()) <= 1.0 + 1e-5)

    def test_softmin(self):
        import lucid.nn.functional as F
        x = lucid.randn(2, 5)
        sm = F.softmin(x, dim=1)
        assert_close(sm.numpy().sum(axis=1), np.ones(2))

    def test_glu(self):
        import lucid.nn.functional as F
        x = lucid.randn(3, 8)
        out = F.glu(x, dim=1)
        assert out.shape == (3, 4)

    def test_unfold(self):
        import lucid.nn.functional as F
        x = lucid.randn(1, 3, 8, 8)
        out = F.unfold(x, kernel_size=3)
        assert out.shape == (1, 27, 36)
