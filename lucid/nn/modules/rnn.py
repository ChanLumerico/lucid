from typing import Literal

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import Numeric, _DeviceType

from .activation import Tanh, ReLU


__all__ = ["RNNCell", "LSTMCell", "GRUCell", "RNNBase", "RNN", "LSTM", "GRU"]


def _get_activation(nonlinearity: str) -> type[nn.Module]:
    if nonlinearity == "tanh":
        return Tanh
    elif nonlinearity == "relu":
        return ReLU
    else:
        raise ValueError(
            f"Invalid nonlinearity '{nonlinearity}'. "
            "Supported nonlinearities are 'tanh' and 'relu'."
        )


class RNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = _get_activation(nonlinearity)()

        sqrt_k = 1.0 / (hidden_size**0.5)
        self.weight_ih = nn.Parameter(
            lucid.random.uniform(-sqrt_k, sqrt_k, (self.hidden_size, self.input_size))
        )
        self.weight_hh = nn.Parameter(
            lucid.random.uniform(-sqrt_k, sqrt_k, (self.hidden_size, self.hidden_size))
        )

        if self.bias:
            self.bias_ih = nn.Parameter(
                lucid.random.uniform(-sqrt_k, sqrt_k, self.hidden_size)
            )
            self.bias_hh = nn.Parameter(
                lucid.random.uniform(-sqrt_k, sqrt_k, self.hidden_size)
            )
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, input_: Tensor, hx: Tensor | None = None) -> Tensor:
        if input_.ndim not in (1, 2):
            raise ValueError(
                "RNNCell expected input with 1 or 2 dimensions, "
                f"got {input_.ndim} dimensions"
            )

        is_batched = input_.ndim == 2
        if not is_batched:
            input_ = input_.unsqueeze(axis=0)
        batch_size = input_.shape[0]

        if hx is None:
            hx = lucid.zeros(
                batch_size, self.hidden_size, dtype=input_.dtype, device=input_.device
            )
        else:
            if hx.ndim not in (1, 2):
                raise ValueError(
                    "RNNCell expected hidden state with 1 or 2 dimensions, "
                    f"got {hx.ndim} dimensions"
                )
            if hx.ndim == 1:
                hx = hx.unsqueeze(axis=0)

            if hx.shape[0] != batch_size or hx.shape[1] != self.hidden_size:
                raise ValueError(
                    "RNNCell expected hidden state with shape "
                    f"({batch_size}, {self.hidden_size}), got {hx.shape}"
                )

        hy = F.linear(input_, self.weight_ih, self.bias_ih)
        hy += F.linear(hx, self.weight_hh, self.bias_hh)
        ret = self.nonlinearity(hy)

        if not is_batched:
            ret = ret.squeeze(axis=0)
        return ret


class LSTMCell(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True, **kwargs
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        sqrt_k = 1.0 / (hidden_size**0.5)
        self.weight_ih = nn.Parameter(
            lucid.random.uniform(
                -sqrt_k, sqrt_k, (4 * self.hidden_size, self.input_size)
            )
        )
        self.weight_hh = nn.Parameter(
            lucid.random.uniform(
                -sqrt_k, sqrt_k, (4 * self.hidden_size, self.hidden_size)
            )
        )

        if self.bias:
            self.bias_ih = nn.Parameter(
                lucid.random.uniform(-sqrt_k, sqrt_k, 4 * self.hidden_size)
            )
            self.bias_hh = nn.Parameter(
                lucid.random.uniform(-sqrt_k, sqrt_k, 4 * self.hidden_size)
            )
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(
        self, input_: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        if input_.ndim not in (1, 2):
            raise ValueError(
                "LSTMCell expected input with 1 or 2 dimensions, "
                f"got {input_.ndim} dimensions"
            )

        is_batched = input_.ndim == 2
        if not is_batched:
            input_ = input_.unsqueeze(axis=0)
        batch_size = input_.shape[0]

        if hx is None:
            h_t = lucid.zeros(
                batch_size, self.hidden_size, dtype=input_.dtype, device=input_.device
            )
            c_t = lucid.zeros(
                batch_size, self.hidden_size, dtype=input_.dtype, device=input_.device
            )
        else:
            h_t, c_t = hx
            if h_t.ndim not in (1, 2) or c_t.ndim not in (1, 2):
                raise ValueError(
                    "LSTMCell expected hidden state and cell state with 1 or 2 dimensions"
                )

            if h_t.ndim == 1:
                h_t = h_t.unsqueeze(axis=0)
            if c_t.ndim == 1:
                c_t = c_t.unsqueeze(axis=0)

            if h_t.shape[0] != batch_size or h_t.shape[1] != self.hidden_size:
                raise ValueError(
                    "LSTMCell expected hidden state with shape "
                    f"({batch_size}, {self.hidden_size}), got {h_t.shape}"
                )
            if c_t.shape[0] != batch_size or c_t.shape[1] != self.hidden_size:
                raise ValueError(
                    "LSTMCell expected cell state with shape "
                    f"({batch_size}, {self.hidden_size}), got {c_t.shape}"
                )

        gates = F.linear(input_, self.weight_ih, self.bias_ih)
        gates += F.linear(h_t, self.weight_hh, self.bias_hh)

        i_t, f_t, g_t, o_t = lucid.split(gates, 4, axis=1)
        i_t = F.sigmoid(i_t)
        f_t = F.sigmoid(f_t)
        g_t = F.tanh(g_t)
        o_t = F.sigmoid(o_t)

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * F.tanh(c_t)

        if not is_batched:
            h_t = h_t.squeeze(axis=0)
            c_t = c_t.squeeze(axis=0)
        return h_t, c_t


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        sqrt_k = 1.0 / (hidden_size**0.5)
        self.weight_ih = nn.Parameter(
            lucid.random.uniform(
                -sqrt_k, sqrt_k, (3 * self.hidden_size, self.input_size)
            )
        )
        self.weight_hh = nn.Parameter(
            lucid.random.uniform(
                -sqrt_k, sqrt_k, (3 * self.hidden_size, self.hidden_size)
            )
        )

        if self.bias:
            self.bias_ih = nn.Parameter(
                lucid.random.uniform(-sqrt_k, sqrt_k, 3 * self.hidden_size)
            )
            self.bias_hh = nn.Parameter(
                lucid.random.uniform(-sqrt_k, sqrt_k, 3 * self.hidden_size)
            )
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, input_: Tensor, hx: Tensor | None = None) -> Tensor:
        if input_.ndim not in (1, 2):
            raise ValueError(
                "GRUCell expected input with 1 or 2 dimensions, "
                f"got {input_.ndim} dimensions"
            )

        is_batched = input_.ndim == 2
        if not is_batched:
            input_ = input_.unsqueeze(axis=0)
        batch_size = input_.shape[0]

        if hx is None:
            hx = lucid.zeros(
                batch_size, self.hidden_size, dtype=input_.dtype, device=input_.device
            )
        else:
            if hx.ndim not in (1, 2):
                raise ValueError(
                    "GRUCell expected hidden state with 1 or 2 dimensions, "
                    f"got {hx.ndim} dimensions"
                )

            if hx.ndim == 1:
                hx = hx.unsqueeze(axis=0)
            if hx.shape[0] != batch_size or hx.shape[1] != self.hidden_size:
                raise ValueError(
                    "GRUCell expected hidden state with shape "
                    f"({batch_size}, {self.hidden_size}), got {hx.shape}"
                )

        input_gates = F.linear(input_, self.weight_ih, self.bias_ih)
        hidden_gates = F.linear(hx, self.weight_hh, self.bias_hh)

        i_r, i_z, i_n = lucid.split(input_gates, 3, axis=1)
        h_r, h_z, h_n = lucid.split(hidden_gates, 3, axis=1)

        r_t = F.sigmoid(i_r + h_r)
        z_t = F.sigmoid(i_z + h_z)
        n_t = F.tanh(i_n + r_t * h_n)

        h_t = (1 - z_t) * n_t + z_t * hx

        if not is_batched:
            h_t = h_t.squeeze(axis=0)
        return h_t


class RNNBase(nn.Module):
    def __init__(
        self,
        mode: Literal["RNN_TANH", "RNN_RELU", "LSTM", "GRU"],
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.is_lstm = False
        cell_kwargs = {}
        nonlinearity = "tanh"

        if mode == "RNN_TANH":
            cell_cls = RNNCell
            cell_kwargs: dict[str, object] = {"nonlinearity": nonlinearity}
        elif mode == "RNN_RELU":
            nonlinearity = "relu"
            cell_cls = RNNCell
            cell_kwargs = {"nonlinearity": nonlinearity}
        elif mode == "LSTM":
            cell_cls = LSTMCell
            self.is_lstm = True
        elif mode == "GRU":
            cell_cls = GRUCell
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes are 'RNN_TANH', "
                "'RNN_RELU', 'LSTM', or 'GRU'."
            )

        self.mode = mode
        self.nonlinearity = nonlinearity

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            layers.append(
                cell_cls(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    bias=bias,
                    **cell_kwargs,
                )
            )
        self.layers = nn.ModuleList(layers)

    def _init_hidden(
        self, batch_size: int, dtype: Numeric, device: _DeviceType
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.is_lstm:
            h0 = lucid.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=dtype,
                device=device,
            )
            c0 = lucid.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=dtype,
                device=device,
            )
            return h0, c0
        return lucid.zeros(
            self.num_layers, batch_size, self.hidden_size, dtype=dtype, device=device
        )

    def forward(
        self, input_: Tensor, hx: Tensor | tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, tuple[Tensor, Tensor]]:
        if input_.ndim != 3:
            raise ValueError(
                f"RNNBase expected input with 3 dimensions, got {input_.ndim} dimensions"
            )

        if self.batch_first:
            input_ = input_.swapaxes(0, 1)

        seq_len, batch_size, feat = input_.shape
        if feat != self.input_size:
            raise ValueError(
                f"RNNBase expected input with feature size {self.input_size}, got {feat}"
            )

        if self.is_lstm:
            if hx is None:
                hx = self._init_hidden(batch_size, input_.dtype, input_.device)
            if not (
                isinstance(hx, (tuple, list))
                and len(hx) == 2
                and isinstance(hx[0], Tensor)
                and isinstance(hx[1], Tensor)
            ):
                raise ValueError("LSTM expects hx as a tuple of (h_0, c_0)")

            h0, c0 = hx
            if h0.ndim == 2:
                h0 = h0.unsqueeze(axis=0)
            if c0.ndim == 2:
                c0 = c0.unsqueeze(axis=0)

            if h0.ndim != 3 or c0.ndim != 3:
                raise ValueError("LSTM expects h_0 and c_0 with 3 dimensions")
            if h0.shape[0] != self.num_layers or c0.shape[0] != self.num_layers:
                raise ValueError("Incorrect number of layers in h_0 or c_0")
            if h0.shape[1] != batch_size or c0.shape[1] != batch_size:
                raise ValueError("Incorrect batch size in h_0 or c_0")
            if h0.shape[2] != self.hidden_size or c0.shape[2] != self.hidden_size:
                raise ValueError("Incorrect hidden size in h_0 or c_0")

            hx_h, hx_c = h0, c0

        else:
            if hx is None:
                hx = self._init_hidden(batch_size, input_.dtype, input_.device)
            if hx.ndim == 2:
                hx = hx.unsqueeze(axis=0)
            if hx.ndim != 3:
                raise ValueError(
                    f"RNNBase expected hidden state with 3 dimensions, got {hx.ndim} dimensions"
                )

            if hx.shape[0] != self.num_layers or hx.shape[1] != batch_size:
                raise ValueError("hx has incorrect shape")
            if hx.shape[2] != self.hidden_size:
                raise ValueError("Incorrect hidden size in hx")

        layer_input = input_
        h_n_list: list[Tensor] = []
        c_n_list: list[Tensor] | None = [] if self.is_lstm else None

        for layer_idx, cell in enumerate(self.layers):
            if self.is_lstm:
                h_t = hx_h[layer_idx]
                c_t = hx_c[layer_idx]
            else:
                h_t = hx[layer_idx]
            outputs = []

            for t in range(seq_len):
                if self.is_lstm:
                    h_t, c_t = cell(layer_input[t], (h_t, c_t))
                    outputs.append(h_t.unsqueeze(axis=0))
                else:
                    h_t = cell(layer_input[t], h_t)
                    outputs.append(h_t.unsqueeze(axis=0))

            layer_output = lucid.concatenate(tuple(outputs), axis=0)

            if self.training and self.dropout > 0.0 and layer_idx < self.num_layers - 1:
                layer_output = F.dropout(layer_output, p=self.dropout)

            h_n_list.append(h_t.unsqueeze(axis=0))
            if self.is_lstm and c_n_list is not None:
                c_n_list.append(c_t.unsqueeze(axis=0))
            layer_input = layer_output

        output = layer_input
        h_n = lucid.concatenate(tuple(h_n_list), axis=0)
        if self.is_lstm and c_n_list is not None:
            c_n = lucid.concatenate(tuple(c_n_list), axis=0)

        if self.batch_first:
            output = output.swapaxes(0, 1)

        if self.is_lstm and c_n_list is not None:
            return output, (h_n, c_n)
        return output, h_n


class RNN(RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: Literal["tanh", "relu"] = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ) -> None:
        if nonlinearity == "tanh":
            mode = "RNN_TANH"
        elif nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(
                f"Invalid nonlinearity '{nonlinearity}'. "
                "Supported nonlinearities are 'tanh' and 'relu'."
            )

        super().__init__(
            mode=mode,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
        )


class LSTM(RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ) -> None:
        mode = "LSTM"
        super().__init__(
            mode=mode,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
        )


class GRU(RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ) -> None:
        mode = "GRU"
        super().__init__(
            mode=mode,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
        )
