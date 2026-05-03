"""
Multi-head attention module.
"""

import math
from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
# F imported lazily inside forward()
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap


class MultiheadAttention(Module):
    """Multi-head scaled dot-product attention."""

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
        batch_first: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        kdim = kdim or embed_dim
        vdim = vdim or embed_dim

        self.in_proj_weight = Parameter(empty(3 * embed_dim, embed_dim, dtype=dtype, device=device))
        self.out_proj_weight = Parameter(empty(embed_dim, embed_dim, dtype=dtype, device=device))
        self.in_proj_bias: Parameter | None = Parameter(empty(3 * embed_dim, dtype=dtype, device=device)) if bias else None
        self.out_proj_bias: Parameter | None = Parameter(empty(embed_dim, dtype=dtype, device=device)) if bias else None
        self._init_weights()

    def _init_weights(self) -> None:
        init.xavier_uniform_(self.in_proj_weight)
        init.xavier_uniform_(self.out_proj_weight)
        if self.in_proj_bias is not None:
            init.zeros_(self.in_proj_bias)
        if self.out_proj_bias is not None:
            init.zeros_(self.out_proj_bias)

    def forward(
        self,
        query: Any,
        key: Any,
        value: Any,
        key_padding_mask: Any = None,
        need_weights: bool = True,
        attn_mask: Any = None,
    ) -> "tuple[Any, Any | None]":
        # Project Q, K, V
        q_w, k_w, v_w = (
            _wrap(_C_engine.reshape(
                _C_engine.contiguous(
                    _C_engine.slice(self.in_proj_weight._impl, 0, i * self.embed_dim, (i + 1) * self.embed_dim, 1)
                ), [self.embed_dim, self.embed_dim]
            ))
            for i in range(3)
        )
        q = F.linear(query, q_w, None)
        k = F.linear(key, k_w, None)
        v = F.linear(value, v_w, None)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout if self.training else 0.0)
        out = F.linear(out, self.out_proj_weight, self.out_proj_bias)
        return out, None
