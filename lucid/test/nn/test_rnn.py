"""Tests for RNN, GRU, LSTM layers."""

import pytest
import lucid
import lucid.nn as nn
from lucid.test.helpers.numerics import make_tensor


class TestRNN:
    def test_output_shape(self):
        rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
        x = make_tensor((4, 10, 8))  # (batch, seq, features)
        out, h = rnn(x)
        assert out.shape == (4, 10, 16)
        assert h.shape == (1, 4, 16)

    def test_bidirectional_shape(self):
        rnn = nn.RNN(8, 16, bidirectional=True, batch_first=True)
        x = make_tensor((4, 10, 8))
        out, h = rnn(x)
        assert out.shape == (4, 10, 32)   # 2 * hidden_size
        assert h.shape == (2, 4, 16)

    def test_multi_layer_shape(self):
        rnn = nn.RNN(8, 16, num_layers=2, batch_first=True)
        x = make_tensor((4, 10, 8))
        out, h = rnn(x)
        assert out.shape == (4, 10, 16)
        assert h.shape == (2, 4, 16)


class TestGRU:
    def test_output_shape(self):
        gru = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
        x = make_tensor((4, 10, 8))
        out, h = gru(x)
        assert out.shape == (4, 10, 16)
        assert h.shape == (1, 4, 16)

    def test_with_initial_hidden(self):
        gru = nn.GRU(8, 16, batch_first=True)
        x = make_tensor((4, 10, 8))
        h0 = make_tensor((1, 4, 16))
        out, h = gru(x, h0)
        assert out.shape == (4, 10, 16)


class TestLSTM:
    def test_output_shape(self):
        lstm = nn.LSTM(input_size=8, hidden_size=16, batch_first=True)
        x = make_tensor((4, 10, 8))
        out, (h, c) = lstm(x)
        assert out.shape == (4, 10, 16)
        assert h.shape == (1, 4, 16)
        assert c.shape == (1, 4, 16)

    def test_multi_layer(self):
        lstm = nn.LSTM(8, 16, num_layers=2, batch_first=True)
        x = make_tensor((4, 10, 8))
        out, (h, c) = lstm(x)
        assert out.shape == (4, 10, 16)
        assert h.shape == (2, 4, 16)

    def test_with_initial_state(self):
        lstm = nn.LSTM(8, 16, batch_first=True)
        x = make_tensor((4, 10, 8))
        h0 = make_tensor((1, 4, 16))
        c0 = make_tensor((1, 4, 16))
        out, (h, c) = lstm(x, (h0, c0))
        assert out.shape == (4, 10, 16)

    def test_backward(self):
        lstm = nn.LSTM(8, 16, batch_first=True)
        x = make_tensor((2, 5, 8), requires_grad=True)
        out, _ = lstm(x)
        lucid.sum(out).backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 5, 8)


class TestRNNCells:
    def test_rnn_cell(self):
        cell = nn.RNNCell(input_size=8, hidden_size=16)
        x = make_tensor((4, 8))
        h = make_tensor((4, 16))
        h_new = cell(x, h)
        assert h_new.shape == (4, 16)

    def test_gru_cell(self):
        cell = nn.GRUCell(input_size=8, hidden_size=16)
        x = make_tensor((4, 8))
        h = make_tensor((4, 16))
        h_new = cell(x, h)
        assert h_new.shape == (4, 16)

    def test_lstm_cell(self):
        cell = nn.LSTMCell(input_size=8, hidden_size=16)
        x = make_tensor((4, 8))
        h = make_tensor((4, 16))
        c = make_tensor((4, 16))
        h_new, c_new = cell(x, (h, c))
        assert h_new.shape == (4, 16)
        assert c_new.shape == (4, 16)
