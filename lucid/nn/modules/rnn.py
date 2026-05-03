"""
Recurrent modules: LSTM, GRU, RNN.
"""

import math
from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap


class LSTM(Module):
    """
    Long short-term memory (LSTM) recurrent layer.

    Input: (seq_len, batch, input_size) or (batch, seq_len, input_size) if batch_first
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        gate_size = 4 * hidden_size

        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                layer_input = input_size if layer == 0 else hidden_size * num_directions
                self.register_parameter(
                    f"weight_ih_l{layer}{suffix}",
                    Parameter(empty(gate_size, layer_input, dtype=dtype, device=device)),
                )
                self.register_parameter(
                    f"weight_hh_l{layer}{suffix}",
                    Parameter(empty(gate_size, hidden_size, dtype=dtype, device=device)),
                )
                if bias:
                    self.register_parameter(
                        f"bias_ih_l{layer}{suffix}",
                        Parameter(empty(gate_size, dtype=dtype, device=device)),
                    )
                    self.register_parameter(
                        f"bias_hh_l{layer}{suffix}",
                        Parameter(empty(gate_size, dtype=dtype, device=device)),
                    )
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(
        self,
        x: Any,
        hx: tuple[Any, Any] | None = None,
    ) -> tuple[Any, tuple[Any, Any]]:
        h0_impl = _unwrap(hx[0]) if hx is not None else None
        c0_impl = _unwrap(hx[1]) if hx is not None else None

        # Collect weights in order: [wih_l0, whh_l0, bih_l0, bhh_l0, ...]
        weights = []
        num_dirs = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            for direction in range(num_dirs):
                suffix = "_reverse" if direction == 1 else ""
                weights.append(_unwrap(self._parameters[f"weight_ih_l{layer}{suffix}"]))
                weights.append(_unwrap(self._parameters[f"weight_hh_l{layer}{suffix}"]))
                if self.bias:
                    weights.append(_unwrap(self._parameters[f"bias_ih_l{layer}{suffix}"]))
                    weights.append(_unwrap(self._parameters[f"bias_hh_l{layer}{suffix}"]))

        # Engine always expects/returns (T, B, H) — transpose if batch_first
        x_impl = _unwrap(x)
        if self.batch_first:
            x_impl = _C_engine.permute(x_impl, [1, 0, 2])

        output_impl, h_n_impl, c_n_impl = _C_engine.nn.lstm_forward(
            x_impl,
            h0_impl, c0_impl,
            weights,
            self.hidden_size,
            self.num_layers,
            False,  # always seq-first to engine
            self.bidirectional,
            self.bias,
        )

        output = _wrap(output_impl)
        if self.batch_first:
            output = _wrap(_C_engine.permute(output._impl, [1, 0, 2]))

        return output, (_wrap(h_n_impl), _wrap(c_n_impl))

    def extra_repr(self) -> str:
        return (f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"bias={self.bias}, batch_first={self.batch_first}, "
                f"dropout={self.dropout_val}, bidirectional={self.bidirectional}")


# ── Pure-Python RNN cells ─────────────────────────────────────────────────────

class RNNCell(Module):
    """Single-step Elman RNN cell: h = tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = "tanh", device: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(empty(hidden_size, input_size, dtype=dtype, device=device))
        self.weight_hh = Parameter(empty(hidden_size, hidden_size, dtype=dtype, device=device))
        self.bias_ih: Parameter | None = Parameter(empty(hidden_size, dtype=dtype, device=device)) if bias else None
        self.bias_hh: Parameter | None = Parameter(empty(hidden_size, dtype=dtype, device=device)) if bias else None
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, x: Any, hx: Any = None) -> Any:
        from lucid.nn import functional as F
        from lucid._factories.creation import zeros
        if hx is None:
            batch = x.shape[0]
            hx = zeros(batch, self.hidden_size)
        pre = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        return F.tanh(pre) if self.nonlinearity == "tanh" else F.relu(pre)

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, nonlinearity={self.nonlinearity!r}"


class LSTMCell(Module):
    """Single-step LSTM cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(empty(4 * hidden_size, input_size, dtype=dtype, device=device))
        self.weight_hh = Parameter(empty(4 * hidden_size, hidden_size, dtype=dtype, device=device))
        self.bias_ih: Parameter | None = Parameter(empty(4 * hidden_size, dtype=dtype, device=device)) if bias else None
        self.bias_hh: Parameter | None = Parameter(empty(4 * hidden_size, dtype=dtype, device=device)) if bias else None
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, x: Any, hx: tuple[Any, Any] | None = None) -> tuple[Any, Any]:
        from lucid.nn import functional as F
        from lucid._factories.creation import zeros
        if hx is None:
            batch = x.shape[0]
            h0 = zeros(batch, self.hidden_size)
            c0 = zeros(batch, self.hidden_size)
        else:
            h0, c0 = hx
        gates = (F.linear(x, self.weight_ih, self.bias_ih)
                 + F.linear(h0, self.weight_hh, self.bias_hh))
        # split into 4 gates
        hs = self.hidden_size
        i_gate = F.sigmoid(gates[:, :hs])
        f_gate = F.sigmoid(gates[:, hs:2*hs])
        g_gate = F.tanh(gates[:, 2*hs:3*hs])
        o_gate = F.sigmoid(gates[:, 3*hs:])
        c1 = f_gate * c0 + i_gate * g_gate
        h1 = o_gate * F.tanh(c1)
        return h1, c1

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}"


class GRUCell(Module):
    """Single-step GRU cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(empty(3 * hidden_size, input_size, dtype=dtype, device=device))
        self.weight_hh = Parameter(empty(3 * hidden_size, hidden_size, dtype=dtype, device=device))
        self.bias_ih: Parameter | None = Parameter(empty(3 * hidden_size, dtype=dtype, device=device)) if bias else None
        self.bias_hh: Parameter | None = Parameter(empty(3 * hidden_size, dtype=dtype, device=device)) if bias else None
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, x: Any, hx: Any = None) -> Any:
        from lucid.nn import functional as F
        from lucid._factories.creation import zeros, ones
        if hx is None:
            hx = zeros(x.shape[0], self.hidden_size)
        hs = self.hidden_size
        gates_x = F.linear(x, self.weight_ih, self.bias_ih)
        gates_h = F.linear(hx, self.weight_hh, self.bias_hh)
        r = F.sigmoid(gates_x[:, :hs] + gates_h[:, :hs])
        z = F.sigmoid(gates_x[:, hs:2*hs] + gates_h[:, hs:2*hs])
        n = F.tanh(gates_x[:, 2*hs:] + r * gates_h[:, 2*hs:])
        h1 = (ones(x.shape[0], hs) - z) * n + z * hx
        return h1

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}"


class GRU(Module):
    """Multi-layer GRU (pure-Python cell-based implementation)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False, dropout: float = 0.0,
                 bidirectional: bool = False, device: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional
        self._cell = GRUCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype)

    def forward(self, x: Any, hx: Any = None) -> tuple[Any, Any]:
        import lucid
        from lucid._factories.creation import zeros
        if self.batch_first:
            perm = [1, 0] + list(range(2, x.ndim))
            x = x.permute(perm)
        T, B = x.shape[0], x.shape[1]
        h = hx if hx is not None else zeros(B, self.hidden_size)
        outputs: list[Any] = []
        for t in range(T):
            h = self._cell(x[t], h)
            outputs.append(h)
        out = lucid.stack(outputs, 0)
        if self.batch_first:
            perm_back = [1, 0] + list(range(2, out.ndim))
            out = out.permute(perm_back)
        return out, h

    def extra_repr(self) -> str:
        return (f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"batch_first={self.batch_first}")


class RNN(Module):
    """Multi-layer Elman RNN (pure-Python cell-based implementation)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = "tanh", bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 device: Any = None, dtype: Any = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self._cell = RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity,
                             device=device, dtype=dtype)

    def forward(self, x: Any, hx: Any = None) -> tuple[Any, Any]:
        import lucid
        from lucid._factories.creation import zeros
        if self.batch_first:
            perm = [1, 0] + list(range(2, x.ndim))
            x = x.permute(perm)
        T, B = x.shape[0], x.shape[1]
        h = hx if hx is not None else zeros(B, self.hidden_size)
        outputs: list[Any] = []
        for t in range(T):
            h = self._cell(x[t], h)
            outputs.append(h)
        out = lucid.stack(outputs, 0)
        if self.batch_first:
            perm_back = [1, 0] + list(range(2, out.ndim))
            out = out.permute(perm_back)
        return out, h

    def extra_repr(self) -> str:
        return (f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"batch_first={self.batch_first}")
