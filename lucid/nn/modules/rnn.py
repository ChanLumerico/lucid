from typing import Literal

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor

from .activation import Tanh, ReLU


__all__ = ["RNNCell"]


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
