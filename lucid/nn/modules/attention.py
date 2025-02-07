import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _Scalar


__all__ = ["ScaledDotProductAttention", "MultiHeadAttention"]


class ScaledDotProductAttention(nn.Module):
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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")

        kdim = kdim if kdim is not None else embed_dim
        vdim = vdim if vdim is not None else embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(vdim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(lucid.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(lucid.zeros(1, 1, embed_dim))
        else:
            self.bias_k = None
            self.bias_v = None

        self.scale: _Scalar = self.head_dim**-0.5

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        N, q_len = query.shape[:2]
        k_len, v_len = key.shape[1], value.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.reshape(N, self.num_heads, q_len, self.head_dim)
        k = k.reshape(N, self.num_heads, k_len, self.head_dim)
        v = v.reshape(N, self.num_heads, v_len, self.head_dim)

        if self.add_bias_kv:
            bias_k = self.bias_k.reshape(1, self.num_heads, 1, self.head_dim).repeat(
                N, axis=0
            )
            bias_v = self.bias_v.reshape(1, self.num_heads, 1, self.head_dim).repeat(
                N, axis=0
            )
            k = lucid.concatenate([k, bias_k], axis=2)
            v = lucid.concatenate([v, bias_v], axis=2)

            if attn_mask is not None:
                attn_mask = lucid.pad(attn_mask, (0, 1))

        if self.add_zero_attn:
            zeros = lucid.zeros(N, self.num_heads, 1, self.head_dim, dtype=q.dtype)
            k = lucid.concatenate([k, zeros], axis=2)
            v = lucid.concatenate([v, zeros], axis=2)

            if attn_mask is not None:
                attn_mask = lucid.pad(attn_mask, (0, 1))

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_mask = (
                key_padding_mask * -1e12
                if attn_mask is None
                else attn_mask + key_padding_mask * -1e12
            )

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout, is_causal=is_causal, scale=self.scale
        )
        attn_output = attn_output.reshape(N, q_len, self.embed_dim)

        output = self.out_proj(attn_output)
        return output
