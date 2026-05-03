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
    ) -> tuple[Any, Any | None]:
        from lucid.nn import functional as F
        # Project Q, K, V by slicing in_proj_weight into 3 equal blocks
        # in_proj_weight shape: (3 * embed_dim, embed_dim)
        d = self.embed_dim
        wt = self.in_proj_weight._impl  # (3*d, d)
        # split_at([d, 2*d], axis=0) → [w[:d], w[d:2d], w[2d:]]
        parts = _C_engine.split_at(wt, [d, 2 * d], 0)
        q_w = _wrap(parts[0])   # (d, d)
        k_w = _wrap(parts[1])   # (d, d)
        v_w = _wrap(parts[2])   # (d, d)

        # split biases if they exist
        if self.in_proj_bias is not None:
            bt = self.in_proj_bias._impl
            b_parts = _C_engine.split_at(bt, [d, 2 * d], 0)
            q_b, k_b, v_b = _wrap(b_parts[0]), _wrap(b_parts[1]), _wrap(b_parts[2])
        else:
            q_b = k_b = v_b = None

        q = F.linear(query, q_w, q_b)
        k = F.linear(key, k_w, k_b)
        v = F.linear(value, v_w, v_b)

        # SDPA expects (B, T, d) — transpose if seq-first (T, B, d)
        if not self.batch_first and q.ndim == 3:
            q = q.permute([1, 0, 2])
            k = k.permute([1, 0, 2])
            v = v.permute([1, 0, 2])

        out = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout if self.training else 0.0)

        # Transpose back to seq-first if needed
        if not self.batch_first and out.ndim == 3:
            out = out.permute([1, 0, 2])

        out = F.linear(out, self.out_proj_weight, self.out_proj_bias)
        return out, None

    def extra_repr(self) -> str:
        return (f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"dropout={self.dropout}, batch_first={self.batch_first}")
