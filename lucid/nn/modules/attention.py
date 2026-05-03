"""
Multi-head attention module.
"""

import math
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.linear import linear
from lucid.nn.functional.attention import scaled_dot_product_attention


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
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        kdim = kdim or embed_dim
        vdim = vdim or embed_dim

        self.in_proj_weight = Parameter(
            empty(3 * embed_dim, embed_dim, dtype=dtype, device=device)
        )
        self.out_proj_weight = Parameter(
            empty(embed_dim, embed_dim, dtype=dtype, device=device)
        )
        self.in_proj_bias: Parameter | None = (
            Parameter(empty(3 * embed_dim, dtype=dtype, device=device))
            if bias
            else None
        )
        self.out_proj_bias: Parameter | None = (
            Parameter(empty(embed_dim, dtype=dtype, device=device)) if bias else None
        )
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
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        d = self.embed_dim
        wt = self.in_proj_weight._impl
        parts = _C_engine.split_at(wt, [d, 2 * d], 0)
        q_w = _wrap(parts[0])
        k_w = _wrap(parts[1])
        v_w = _wrap(parts[2])

        if self.in_proj_bias is not None:
            bt = self.in_proj_bias._impl
            b_parts = _C_engine.split_at(bt, [d, 2 * d], 0)
            q_b, k_b, v_b = _wrap(b_parts[0]), _wrap(b_parts[1]), _wrap(b_parts[2])
        else:
            q_b = k_b = v_b = None

        q = linear(query, q_w, q_b)
        k = linear(key, k_w, k_b)
        v = linear(value, v_w, v_b)

        if not self.batch_first and q.ndim == 3:
            q = q.permute([1, 0, 2])
            k = k.permute([1, 0, 2])
            v = v.permute([1, 0, 2])

        out = scaled_dot_product_attention(
            q, k, v, attn_mask, self.dropout if self.training else 0.0
        )

        if not self.batch_first and out.ndim == 3:
            out = out.permute([1, 0, 2])

        out = linear(out, self.out_proj_weight, self.out_proj_bias)
        return out, None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout}, batch_first={self.batch_first}"
        )
