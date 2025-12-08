from typing import Literal

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import Numeric, _DeviceType

from .activation import Tanh, ReLU


__all__ = ["RNNCell", "RNNBase"]


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


class RNNBase(nn.Module):
    def __init__(
        self,
        mode: Literal["RNN_TANH", "RNN_RELU"],
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode == "RNN_TANH":
            nonlinearity = "tanh"
            cell_cls = RNNCell
        elif mode == "RNN_RELU":
            nonlinearity = "relu"
            cell_cls = RNNCell
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes are 'RNN_TANH' and 'RNN_RELU'."
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
                    nonlinearity=nonlinearity,
                )
            )
        self.layers = nn.ModuleList(layers)

    def _init_hidden(
        self, batch_size: int, dtype: Numeric, device: _DeviceType
    ) -> Tensor:
        return lucid.zeros(
            self.num_layers, batch_size, self.hidden_size, dtype=dtype, device=device
        )

    def forward(
        self, input_: Tensor, hx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
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

        if hx is None:
            hx = self._init_hidden(batch_size, input_.dtype, input_.device)
        else:
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

        for layer_idx, cell in enumerate(self.layers):
            h_t = hx[layer_idx]
            outputs = []

            for t in range(seq_len):
                h_t = cell(layer_input[t], h_t)
                outputs.append(h_t.unsqueeze(axis=0))

            layer_output = lucid.concatenate(tuple(outputs), axis=0)
            if self.training and self.dropout > 0.0 and layer_idx < self.num_layers - 1:
                layer_output = F.dropout(layer_output, p=self.dropout)

            h_n_list.append(h_t.unsqueeze(axis=0))
            layer_input = layer_output

        output = layer_input
        h_n = lucid.concatenate(tuple(h_n_list), axis=0)

        if self.batch_first:
            output = output.swapaxes(0, 1)

        return output, h_n
