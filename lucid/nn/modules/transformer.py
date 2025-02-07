from typing import Callable

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["TransformerEncoderLayer", "TransformerDecoderLayer"]


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(
            d_model, num_heads, dropout=dropout, bias=bias
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = activation
        self.norm_first = norm_first

    def _sa_block(
        self,
        x: Tensor,
        src_mask: Tensor | None,
        src_key_padding_mask: Tensor,
        is_causal: bool,
    ) -> Tensor:
        attn_output = self.self_attn(x, x, x, src_key_padding_mask, src_mask, is_causal)
        attn_output = self.dropout1(attn_output)

        return attn_output

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            x = src + self._sa_block(
                self.norm1(src), src_mask, src_key_padding_mask, is_causal
            )
            x += self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                src + self._sa_block(src, src_mask, src_key_padding_mask, is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    NotImplemented
