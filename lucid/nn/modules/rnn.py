"""
Recurrent modules: LSTM, GRU, RNN.
"""

import math
from typing import TYPE_CHECKING

from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty, zeros
from lucid._factories.random import rand
import lucid.nn.init as init
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.linear import linear
from lucid.nn.functional.activations import tanh, relu, sigmoid
from lucid._ops import stack

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cat_last(a: "Tensor", b: "Tensor") -> "Tensor":
    """Concatenate two tensors along the last dimension."""
    import lucid
    return lucid.cat([a, b], a.ndim - 1)


# ── LSTM ──────────────────────────────────────────────────────────────────────


class LSTM(Module):
    """Long Short-Term Memory recurrent layer.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input.
    hidden_size : int
        Number of features in the hidden state.
    num_layers : int, optional
        Number of recurrent layers (default: 1).
    bias : bool, optional
        If ``False``, the layer does not use bias weights (default: ``True``).
        Due to an engine limitation, a zero-valued bias is still passed to the
        engine; numerically equivalent to no bias, but bias tensors exist.
    batch_first : bool, optional
        If ``True``, input/output tensors are ``(batch, seq, feature)``
        instead of ``(seq, batch, feature)`` (default: ``False``).
    dropout : float, optional
        Dropout probability between LSTM layers (default: 0.0).
    bidirectional : bool, optional
        If ``True``, use a bidirectional LSTM (default: ``False``).

    Notes
    -----
    The output hidden state ``h_n`` has shape
    ``(D * num_layers, batch, hidden_size)`` where ``D = 2`` if bidirectional
    else ``D = 1``.  This matches PyTorch convention.

    Examples
    --------
    >>> lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
    >>> x = lucid.randn(2, 5, 10)
    >>> output, (h_n, c_n) = lstm(x)
    >>> output.shape
    (2, 5, 20)
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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
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
        x: "Tensor",
        hx: "tuple[Tensor, Tensor] | None" = None,
    ) -> "tuple[Tensor, tuple[Tensor, Tensor]]":
        h0_impl = _unwrap(hx[0]) if hx is not None else None
        c0_impl = _unwrap(hx[1]) if hx is not None else None

        num_dirs = 2 if self.bidirectional else 1
        gate_size = 4 * self.hidden_size

        weights = []
        for layer in range(self.num_layers):
            for direction in range(num_dirs):
                suffix = "_reverse" if direction == 1 else ""
                weights.append(_unwrap(self._parameters[f"weight_ih_l{layer}{suffix}"]))
                weights.append(_unwrap(self._parameters[f"weight_hh_l{layer}{suffix}"]))
                if self.bias:
                    weights.append(_unwrap(self._parameters[f"bias_ih_l{layer}{suffix}"]))
                    weights.append(_unwrap(self._parameters[f"bias_hh_l{layer}{suffix}"]))
                else:
                    # Engine training path requires bias tensors; supply zeros
                    # so computation is equivalent to bias=False.
                    dev = _unwrap(x).device
                    dt = _unwrap(x).dtype
                    zero_b = _C_engine.TensorImpl(
                        __import__("numpy").zeros(gate_size, dtype="float32"),
                        dev, False
                    )
                    weights.append(zero_b)
                    weights.append(zero_b)

        x_impl = _unwrap(x)
        if self.batch_first:
            x_impl = _C_engine.permute(x_impl, [1, 0, 2])

        output_impl, h_n_impl, c_n_impl = _C_engine.nn.lstm_forward(
            x_impl,
            h0_impl,
            c0_impl,
            weights,
            self.hidden_size,
            self.num_layers,
            False,
            self.bidirectional,
            True,   # always True: we always supply bias tensors above
        )

        output = _wrap(output_impl)
        if self.batch_first:
            output = _wrap(_C_engine.permute(output._impl, [1, 0, 2]))

        return output, (_wrap(h_n_impl), _wrap(c_n_impl))

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"bias={self.bias}, batch_first={self.batch_first}, "
            f"dropout={self.dropout_val}, bidirectional={self.bidirectional}"
        )


# ── Pure-Python RNN cells ─────────────────────────────────────────────────────


class RNNCell(Module):
    """Single-step Elman RNN cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(
            empty(hidden_size, input_size, dtype=dtype, device=device)
        )
        self.weight_hh = Parameter(
            empty(hidden_size, hidden_size, dtype=dtype, device=device)
        )
        self.bias_ih: Parameter | None = (
            Parameter(empty(hidden_size, dtype=dtype, device=device)) if bias else None
        )
        self.bias_hh: Parameter | None = (
            Parameter(empty(hidden_size, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, x: "Tensor", hx: "Tensor | None" = None) -> "Tensor":
        if hx is None:
            hx = zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        pre = linear(x, self.weight_ih, self.bias_ih) + linear(
            hx, self.weight_hh, self.bias_hh
        )
        return tanh(pre) if self.nonlinearity == "tanh" else relu(pre)

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, nonlinearity={self.nonlinearity!r}"
        )


class LSTMCell(Module):
    """Single-step LSTM cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(
            empty(4 * hidden_size, input_size, dtype=dtype, device=device)
        )
        self.weight_hh = Parameter(
            empty(4 * hidden_size, hidden_size, dtype=dtype, device=device)
        )
        self.bias_ih: Parameter | None = (
            Parameter(empty(4 * hidden_size, dtype=dtype, device=device))
            if bias else None
        )
        self.bias_hh: Parameter | None = (
            Parameter(empty(4 * hidden_size, dtype=dtype, device=device))
            if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(
        self,
        x: "Tensor",
        hx: "tuple[Tensor, Tensor] | None" = None,
    ) -> "tuple[Tensor, Tensor]":
        if hx is None:
            batch = x.shape[0]
            h0 = zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
            c0 = zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h0, c0 = hx
        gates = linear(x, self.weight_ih, self.bias_ih) + linear(
            h0, self.weight_hh, self.bias_hh
        )
        hs = self.hidden_size
        i_gate = sigmoid(gates[:, :hs])
        f_gate = sigmoid(gates[:, hs : 2 * hs])
        g_gate = tanh(gates[:, 2 * hs : 3 * hs])
        o_gate = sigmoid(gates[:, 3 * hs :])
        c1 = f_gate * c0 + i_gate * g_gate
        h1 = o_gate * tanh(c1)
        return h1, c1

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}"


class GRUCell(Module):
    """Single-step GRU cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(
            empty(3 * hidden_size, input_size, dtype=dtype, device=device)
        )
        self.weight_hh = Parameter(
            empty(3 * hidden_size, hidden_size, dtype=dtype, device=device)
        )
        self.bias_ih: Parameter | None = (
            Parameter(empty(3 * hidden_size, dtype=dtype, device=device))
            if bias else None
        )
        self.bias_hh: Parameter | None = (
            Parameter(empty(3 * hidden_size, dtype=dtype, device=device))
            if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, x: "Tensor", hx: "Tensor | None" = None) -> "Tensor":
        if hx is None:
            hx = zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        hs = self.hidden_size
        gates_x = linear(x, self.weight_ih, self.bias_ih)
        gates_h = linear(hx, self.weight_hh, self.bias_hh)
        r = sigmoid(gates_x[:, :hs] + gates_h[:, :hs])
        z = sigmoid(gates_x[:, hs : 2 * hs] + gates_h[:, hs : 2 * hs])
        n = tanh(gates_x[:, 2 * hs :] + r * gates_h[:, 2 * hs :])
        h1 = (1.0 - z) * n + z * hx
        return h1

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}"


# ── Multi-step GRU ────────────────────────────────────────────────────────────


class GRU(Module):
    """Multi-layer GRU.

    Returns ``(output, h_n)`` where ``h_n`` has shape
    ``(D * num_layers, batch, hidden_size)`` — matching PyTorch convention.

    Parameters
    ----------
    input_size, hidden_size, num_layers, bias, batch_first, dropout,
    bidirectional : same as ``torch.nn.GRU``.

    Examples
    --------
    >>> gru = nn.GRU(8, 16, num_layers=2, batch_first=True)
    >>> x = lucid.randn(2, 5, 8)
    >>> out, h_n = gru(x)
    >>> out.shape, h_n.shape
    ((2, 5, 16), (2, 2, 16))
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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional

        num_dirs = 2 if bidirectional else 1
        for layer in range(num_layers):
            for d in range(num_dirs):
                suffix = "_reverse" if d == 1 else ""
                in_sz = input_size if layer == 0 else hidden_size * num_dirs
                self.add_module(
                    f"cell_l{layer}{suffix}",
                    GRUCell(in_sz, hidden_size, bias=bias, device=device, dtype=dtype),
                )

    def _cell(self, layer: int, reverse: bool = False) -> GRUCell:
        suffix = "_reverse" if reverse else ""
        return self._modules[f"cell_l{layer}{suffix}"]  # type: ignore[return-value]

    def forward(
        self,
        x: "Tensor",
        hx: "Tensor | None" = None,
    ) -> "tuple[Tensor, Tensor]":
        if self.batch_first:
            x = x.permute([1, 0, 2])

        T, B = x.shape[0], x.shape[1]
        num_dirs = 2 if self.bidirectional else 1

        if hx is None:
            hx = zeros(
                self.num_layers * num_dirs, B, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h_n: list[Tensor] = []  # type: ignore[name-defined]
        inp = x

        for layer in range(self.num_layers):
            # ── forward direction ────────────────────────────────────────────
            cell_fwd = self._cell(layer, reverse=False)
            h_fwd = hx[layer * num_dirs]  # (B, hidden)
            fwd_out: list[Tensor] = []  # type: ignore[name-defined]
            for t in range(T):
                h_fwd = cell_fwd(inp[t], h_fwd)
                fwd_out.append(h_fwd)
            h_n.append(h_fwd)

            if self.bidirectional:
                # ── reverse direction ────────────────────────────────────────
                cell_rev = self._cell(layer, reverse=True)
                h_rev = hx[layer * num_dirs + 1]
                rev_out: list[Tensor] = []  # type: ignore[name-defined]
                for t in range(T - 1, -1, -1):
                    h_rev = cell_rev(inp[t], h_rev)
                    rev_out.append(h_rev)
                rev_out.reverse()
                h_n.append(h_rev)
                # Concat forward and backward at each time step
                layer_out = stack(
                    [_cat_last(fwd_out[t], rev_out[t]) for t in range(T)], 0
                )
            else:
                layer_out = stack(fwd_out, 0)

            inp = layer_out

        out = inp
        h_n_tensor = stack(h_n, 0)  # (D*num_layers, B, hidden)

        if self.batch_first:
            out = out.permute([1, 0, 2])

        return out, h_n_tensor

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, bidirectional={self.bidirectional}"
        )


# ── Multi-step RNN ────────────────────────────────────────────────────────────


class RNN(Module):
    """Multi-layer Elman RNN.

    Returns ``(output, h_n)`` where ``h_n`` has shape
    ``(D * num_layers, batch, hidden_size)`` — matching PyTorch convention.

    Parameters
    ----------
    input_size, hidden_size, num_layers, nonlinearity, bias, batch_first,
    dropout, bidirectional : same as ``torch.nn.RNN``.

    Examples
    --------
    >>> rnn = nn.RNN(8, 16, num_layers=2, batch_first=True)
    >>> x = lucid.randn(2, 5, 8)
    >>> out, h_n = rnn(x)
    >>> out.shape, h_n.shape
    ((2, 5, 16), (2, 2, 16))
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        num_dirs = 2 if bidirectional else 1
        for layer in range(num_layers):
            for d in range(num_dirs):
                suffix = "_reverse" if d == 1 else ""
                in_sz = input_size if layer == 0 else hidden_size * num_dirs
                self.add_module(
                    f"cell_l{layer}{suffix}",
                    RNNCell(
                        in_sz, hidden_size, bias=bias,
                        nonlinearity=nonlinearity,
                        device=device, dtype=dtype,
                    ),
                )

    def _cell(self, layer: int, reverse: bool = False) -> RNNCell:
        suffix = "_reverse" if reverse else ""
        return self._modules[f"cell_l{layer}{suffix}"]  # type: ignore[return-value]

    def forward(
        self,
        x: "Tensor",
        hx: "Tensor | None" = None,
    ) -> "tuple[Tensor, Tensor]":
        if self.batch_first:
            x = x.permute([1, 0, 2])

        T, B = x.shape[0], x.shape[1]
        num_dirs = 2 if self.bidirectional else 1

        if hx is None:
            hx = zeros(
                self.num_layers * num_dirs, B, self.hidden_size,
                device=x.device, dtype=x.dtype,
            )

        h_n: list[Tensor] = []  # type: ignore[name-defined]
        inp = x

        for layer in range(self.num_layers):
            cell_fwd = self._cell(layer, reverse=False)
            h_fwd = hx[layer * num_dirs]
            fwd_out: list[Tensor] = []  # type: ignore[name-defined]
            for t in range(T):
                h_fwd = cell_fwd(inp[t], h_fwd)
                fwd_out.append(h_fwd)
            h_n.append(h_fwd)

            if self.bidirectional:
                cell_rev = self._cell(layer, reverse=True)
                h_rev = hx[layer * num_dirs + 1]
                rev_out: list[Tensor] = []  # type: ignore[name-defined]
                for t in range(T - 1, -1, -1):
                    h_rev = cell_rev(inp[t], h_rev)
                    rev_out.append(h_rev)
                rev_out.reverse()
                h_n.append(h_rev)
                layer_out = stack(
                    [_cat_last(fwd_out[t], rev_out[t]) for t in range(T)], 0
                )
            else:
                layer_out = stack(fwd_out, 0)

            inp = layer_out

        out = inp
        h_n_tensor = stack(h_n, 0)

        if self.batch_first:
            out = out.permute([1, 0, 2])

        return out, h_n_tensor

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"nonlinearity={self.nonlinearity!r}, batch_first={self.batch_first}, "
            f"bidirectional={self.bidirectional}"
        )
