import math

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["Transformer"]


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = lucid.zeros(max_len, d_model)
        position = lucid.arange(0, max_len).unsqueeze(axis=1)
        div_term = lucid.exp(lucid.arange(0, d_model, 2) * (-math.log(1e4) / d_model))

        pe[:, 0::2] = lucid.sin(position * div_term)
        pe[:, 1::2] = lucid.cos(position * div_term)

        pe = pe.unsqueeze(axis=1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x += self.pe[: x.shape[0]]
        x = self.dropout(x)

        return x
