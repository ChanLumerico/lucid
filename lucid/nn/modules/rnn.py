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
        # Collect weight parameters
        weight_ih = self._parameters.get("weight_ih_l0")
        weight_hh = self._parameters.get("weight_hh_l0")
        bias_ih = self._parameters.get("bias_ih_l0")
        bias_hh = self._parameters.get("bias_hh_l0")

        assert weight_ih is not None and weight_hh is not None
        bih = _unwrap(bias_ih) if bias_ih is not None else None
        bhh = _unwrap(bias_hh) if bias_hh is not None else None

        output_impl, h_n_impl, c_n_impl = _C_engine.nn.lstm_forward(
            _unwrap(x),
            _unwrap(weight_ih), _unwrap(weight_hh),
            bih, bhh,
            self.batch_first,
        )
        return _wrap(output_impl), (_wrap(h_n_impl), _wrap(c_n_impl))

    def extra_repr(self) -> str:
        return (f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"bias={self.bias}, batch_first={self.batch_first}, "
                f"dropout={self.dropout_val}, bidirectional={self.bidirectional}")
