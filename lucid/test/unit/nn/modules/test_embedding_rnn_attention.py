"""Embedding / RNN / containers / shape modules / padding / transformer."""

import numpy as np

import lucid
import lucid.nn as nn


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
