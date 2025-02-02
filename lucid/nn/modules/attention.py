import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _Scalar


__all__ = ["ScaledDotProductAttention", "MultiHeadAttention"]


class ScaledDotProductAttention(nn.Module):  # NOTE: Undocumented
    def __init__(
        self,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: _Scalar | None = None,
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.scale = scale

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            self.attn_mask,
            self.dropout_p,
            self.is_causal,
            self.scale,
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
    ) -> None:
        super().__init__()

        # TODO: Implement this
