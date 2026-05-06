"""Tests for RNN, GRU, LSTM layers."""

import pytest
import numpy as np

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
        assert out.shape == (4, 10, 32)  # 2 * hidden_size
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


class TestLSTMProjSize:
    def test_proj_size_output_shape(self):
        m = nn.LSTM(input_size=8, hidden_size=16, proj_size=4)
        x = make_tensor((5, 2, 8))
        out, (hn, cn) = m(x)
        # output / h_n shrink to proj_size; c_n stays hidden_size.
        assert out.shape == (5, 2, 4)
        assert hn.shape == (1, 2, 4)
        assert cn.shape == (1, 2, 16)

    def test_proj_size_weight_shapes(self):
        m = nn.LSTM(input_size=8, hidden_size=16, proj_size=4)
        # W_ih: (4*H, input_size) — unchanged.
        assert m.weight_ih_l0.shape == (64, 8)
        # W_hh: (4*H, proj_size) — recurrent dim shrinks.
        assert m.weight_hh_l0.shape == (64, 4)
        # W_hr: (proj_size, hidden_size).
        assert m.weight_hr_l0.shape == (4, 16)

    def test_proj_size_backward_grads(self):
        m = nn.LSTM(input_size=4, hidden_size=8, proj_size=3)
        x = lucid.tensor(
            np.random.default_rng(0).standard_normal((4, 2, 4)).astype(np.float32),
            requires_grad=True,
        )
        out, _ = m(x)
        out.sum().backward()
        assert x.grad is not None
        assert m.weight_ih_l0.grad is not None
        assert m.weight_hh_l0.grad is not None
        assert m.weight_hr_l0.grad is not None
        assert m.weight_ih_l0.grad.shape == m.weight_ih_l0.shape
        assert m.weight_hh_l0.grad.shape == m.weight_hh_l0.shape
        assert m.weight_hr_l0.grad.shape == m.weight_hr_l0.shape

    def test_proj_size_with_multilayer_supported(self):
        # The Python multi-layer driver composes single-layer engine
        # calls, so proj_size now works with stacked layers.
        m = nn.LSTM(8, 16, num_layers=2, proj_size=4)
        x = make_tensor((5, 2, 8))
        out, (hn, cn) = m(x)
        # Output / hn carry the projected dim; cn keeps the cell-state dim.
        assert out.shape == (5, 2, 4)
        assert hn.shape == (2, 2, 4)
        assert cn.shape == (2, 2, 16)

    def test_proj_size_with_bidirectional_supported(self):
        m = nn.LSTM(8, 16, num_layers=1, bidirectional=True, proj_size=4)
        x = make_tensor((5, 2, 8))
        out, (hn, cn) = m(x)
        # Bidirectional concatenates forward + reverse → 2 * proj_size.
        assert out.shape == (5, 2, 8)
        assert hn.shape == (2, 2, 4)
        assert cn.shape == (2, 2, 16)

    def test_proj_size_validation(self):
        with pytest.raises(ValueError, match="proj_size"):
            nn.LSTM(8, 16, proj_size=-1)
        # proj_size >= hidden_size is a degenerate case the reference
        # framework also rejects.
        with pytest.raises(ValueError, match="proj_size"):
            nn.LSTM(8, 16, proj_size=16)

    def test_proj_size_in_repr(self):
        m = nn.LSTM(8, 16, proj_size=4)
        assert "proj_size=4" in repr(m)
        m_default = nn.LSTM(8, 16)
        assert "proj_size" not in repr(m_default)

    def test_proj_size_zero_is_standard_lstm(self):
        # proj_size=0 must take the original path with no hr params.
        m = nn.LSTM(8, 16, proj_size=0)
        assert "weight_hr_l0" not in m._parameters
        x = make_tensor((5, 2, 8))
        out, _ = m(x)
        assert out.shape == (5, 2, 16)


class TestRNNExtras:
    @pytest.mark.parametrize("cls", [nn.LSTM, nn.GRU, nn.RNN])
    def test_flatten_parameters_no_op(self, cls):
        m = cls(8, 16)
        # Just confirm the method exists and returns None.
        assert m.flatten_parameters() is None

    @pytest.mark.parametrize(
        "cls,name", [(nn.LSTM, "LSTM"), (nn.GRU, "GRU"), (nn.RNN, "RNN")]
    )
    def test_packed_sequence_input_rejected(self, cls, name):
        from lucid.nn.utils.rnn import PackedSequence

        m = cls(4, 8)
        ps = PackedSequence(
            data=lucid.zeros(3, 4),
            batch_sizes=lucid.tensor([2, 1]).to(dtype=lucid.int64),
            sorted_indices=None,
            unsorted_indices=None,
        )
        with pytest.raises(NotImplementedError, match=name):
            m(ps)


class TestStackedBidirectional:
    """Multi-layer × bidirectional combinations — coverage gap from Phase 1."""

    @pytest.mark.parametrize("cls", [nn.GRU, nn.RNN])
    def test_stacked_bidirectional_shapes_pure_python(self, cls):
        # GRU/RNN run their bidirectional pass in Python and produce
        # the full (D*num_layers, B, H) hidden stack.
        m = cls(8, 16, num_layers=2, bidirectional=True, batch_first=True)
        x = make_tensor((4, 6, 8))
        y, hn = m(x)
        assert y.shape == (4, 6, 16 * 2)
        assert hn.shape == (2 * 2, 4, 16)

    def test_lstm_single_layer_bidirectional(self):
        m = nn.LSTM(8, 16, num_layers=1, bidirectional=True, batch_first=True)
        x = make_tensor((4, 6, 8))
        y, (hn, cn) = m(x)
        # Concatenated forward + reverse along the last dim.
        assert y.shape == (4, 6, 32)
        # Per-direction layer states: (D*L, B, H).
        assert hn.shape == (2, 4, 16)
        assert cn.shape == (2, 4, 16)

    def test_lstm_multi_layer_bidirectional(self):
        m = nn.LSTM(8, 16, num_layers=2, bidirectional=True, batch_first=True)
        x = make_tensor((4, 6, 8))
        y, (hn, cn) = m(x)
        assert y.shape == (4, 6, 32)
        assert hn.shape == (4, 4, 16)
        assert cn.shape == (4, 4, 16)


class TestGRURNNStateDictNaming:
    @pytest.mark.parametrize("cls", [nn.GRU, nn.RNN])
    def test_state_dict_uses_flat_keys(self, cls):
        m = cls(8, 16, num_layers=2, bidirectional=True)
        sd = m.state_dict()
        # Expect reference-framework naming: weight_ih_l0, weight_hh_l1_reverse, ...
        assert "weight_ih_l0" in sd
        assert "weight_hh_l1_reverse" in sd
        assert "bias_ih_l0_reverse" in sd
        # No legacy cell-prefix keys in the saved dict.
        assert all(not k.startswith("cell_l") for k in sd)

    @pytest.mark.parametrize("cls", [nn.GRU, nn.RNN])
    def test_legacy_v1_keys_still_loadable(self, cls):
        from collections import OrderedDict

        src = cls(8, 16, num_layers=2, bidirectional=True)
        # Hand-build the legacy ``cell_l*.weight_*`` layout from src's params.
        legacy = OrderedDict()
        for layer in range(2):
            for d in range(2):
                suffix = "_reverse" if d == 1 else ""
                cell = src._modules[f"cell_l{layer}{suffix}"]
                for pname in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    p = getattr(cell, pname)
                    if p is not None:
                        legacy[f"cell_l{layer}{suffix}.{pname}"] = p.detach()
        legacy._metadata = {}  # no version → migration path
        dst = cls(8, 16, num_layers=2, bidirectional=True)
        result = dst.load_state_dict(legacy)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        # Sanity-check the parameters actually transferred.
        np.testing.assert_allclose(
            dst.cell_l0.weight_ih.numpy(), src.cell_l0.weight_ih.numpy()
        )

    @pytest.mark.parametrize("cls", [nn.GRU, nn.RNN])
    def test_v2_round_trip(self, cls):
        src = cls(8, 16, num_layers=2, bidirectional=True)
        dst = cls(8, 16, num_layers=2, bidirectional=True)
        dst.load_state_dict(src.state_dict())
        np.testing.assert_allclose(
            dst.cell_l1_reverse.weight_hh.numpy(),
            src.cell_l1_reverse.weight_hh.numpy(),
        )
