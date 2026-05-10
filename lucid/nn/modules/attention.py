"""
Multi-head attention module.
"""

import math
from typing import TYPE_CHECKING, cast

from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
from lucid._dispatch import _wrap
from lucid._C import engine as _C_engine
import lucid as _lucid
import lucid.nn.init as init
from lucid.nn.functional.linear import linear
from lucid.nn.functional.attention import scaled_dot_product_attention

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_NEG_INF: float = float("-inf")


def _to_additive_mask(mask: Tensor, float_dtype: object) -> Tensor:
    """Convert a bool/byte mask (True = mask out) to an additive float mask
    (-inf where True, 0 where False).  Already-float masks pass through."""
    if mask.dtype is _lucid.bool_:
        # mask: True ⇒ -inf, False ⇒ 0
        _dtype = cast(DTypeLike, float_dtype)
        zero_t: Tensor = _lucid.zeros(mask.shape, dtype=_dtype, device=mask.device)
        ninf_t: Tensor = _lucid.full(
            mask.shape, _NEG_INF, dtype=_dtype, device=mask.device
        )
        return _lucid.where(mask, ninf_t, zero_t)
    return mask


class MultiheadAttention(Module):
    """Multi-head scaled dot-product attention.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model (split across heads).
    num_heads : int
        Number of parallel attention heads; ``embed_dim`` must be divisible.
    dropout : float
        Probability applied to attention weights during training.
    bias : bool
        Add a learnable bias to the input / output projections.
    add_bias_kv : bool
        Prepend learnable bias rows to ``K`` and ``V`` along the sequence axis.
    add_zero_attn : bool
        Append a zero row to ``K`` and ``V`` along the sequence axis.
    kdim, vdim : int | None
        Separate dimensions for the key / value inputs.  When unset they
        fall back to ``embed_dim`` and the three projections share a fused
        weight (``in_proj_weight``).
    batch_first : bool
        If ``True``, inputs / outputs are ``(batch, seq, feature)`` rather
        than the default ``(seq, batch, feature)``.
    """

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
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.dropout: float = dropout
        self.batch_first: bool = batch_first
        self.head_dim: int = embed_dim // num_heads
        self.kdim: int = kdim if kdim is not None else embed_dim
        self.vdim: int = vdim if vdim is not None else embed_dim
        self.add_zero_attn: bool = add_zero_attn
        self._qkv_same_embed_dim: bool = (
            self.kdim == embed_dim and self.vdim == embed_dim
        )

        if self._qkv_same_embed_dim:
            self.in_proj_weight: Parameter | None = Parameter(
                empty(3 * embed_dim, embed_dim, dtype=dtype, device=device)
            )
            self.q_proj_weight: Parameter | None = None
            self.k_proj_weight: Parameter | None = None
            self.v_proj_weight: Parameter | None = None
        else:
            # Reference layout: separate projection weights when Q/K/V come
            # from sources with different feature dims.  ``in_proj_weight``
            # is unset; load hooks may still accept it for compatibility.
            self.in_proj_weight = None
            self.q_proj_weight = Parameter(
                empty(embed_dim, embed_dim, dtype=dtype, device=device)
            )
            self.k_proj_weight = Parameter(
                empty(embed_dim, self.kdim, dtype=dtype, device=device)
            )
            self.v_proj_weight = Parameter(
                empty(embed_dim, self.vdim, dtype=dtype, device=device)
            )

        if bias:
            self.in_proj_bias: Parameter | None = Parameter(
                empty(3 * embed_dim, dtype=dtype, device=device)
            )
        else:
            self.in_proj_bias = None

        if add_bias_kv:
            self.bias_k: Parameter | None = Parameter(
                empty(1, 1, embed_dim, dtype=dtype, device=device)
            )
            self.bias_v: Parameter | None = Parameter(
                empty(1, 1, embed_dim, dtype=dtype, device=device)
            )
        else:
            self.bias_k = None
            self.bias_v = None

        # Output projection — kept as flat parameters (``out_proj_weight``
        # / ``out_proj_bias``) for backwards compatibility with checkpoints
        # saved by earlier Lucid versions.  External reference checkpoints
        # use ``out_proj.weight`` / ``out_proj.bias``; the load hook below
        # remaps either form.
        self.out_proj_weight: Parameter = Parameter(
            empty(embed_dim, embed_dim, dtype=dtype, device=device)
        )
        self.out_proj_bias: Parameter | None = (
            Parameter(empty(embed_dim, dtype=dtype, device=device)) if bias else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        if self.in_proj_weight is not None:
            init.xavier_uniform_(self.in_proj_weight)
        if self.q_proj_weight is not None:
            init.xavier_uniform_(self.q_proj_weight)
            assert self.k_proj_weight is not None
            assert self.v_proj_weight is not None
            init.xavier_uniform_(self.k_proj_weight)
            init.xavier_uniform_(self.v_proj_weight)
        init.xavier_uniform_(self.out_proj_weight)
        if self.in_proj_bias is not None:
            init.zeros_(self.in_proj_bias)
        if self.out_proj_bias is not None:
            init.zeros_(self.out_proj_bias)
        if self.bias_k is not None:
            init.xavier_normal_(self.bias_k)
            assert self.bias_v is not None
            init.xavier_normal_(self.bias_v)

    # ── reference-checkpoint loading: accept ``out_proj.weight`` / ``out_proj.bias``
    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Inline-rename ``out_proj.weight`` → ``out_proj_weight`` (and bias)
        # so external checkpoints using the reference framework's MHA
        # layout load directly.  Both forms are tolerated.
        for src_name, dst_name in (
            ("out_proj.weight", "out_proj_weight"),
            ("out_proj.bias", "out_proj_bias"),
        ):
            src_key: str = f"{prefix}{src_name}"
            dst_key: str = f"{prefix}{dst_name}"
            if src_key in state_dict and dst_key not in state_dict:
                state_dict[dst_key] = state_dict.pop(src_key)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _project_qkv(
        self, query: Tensor, key: Tensor, value: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the input projections, returning ``(q, k, v)`` each shaped
        ``(B, T*, embed_dim)``."""
        d: int = self.embed_dim
        if self._qkv_same_embed_dim:
            wt = self.in_proj_weight._impl  # type: ignore[union-attr]
            parts = _C_engine.split_at(wt, [d, 2 * d], 0)
            q_w: Tensor = _wrap(parts[0])
            k_w: Tensor = _wrap(parts[1])
            v_w: Tensor = _wrap(parts[2])
        else:
            q_w = self.q_proj_weight  # type: ignore[assignment]
            k_w = self.k_proj_weight  # type: ignore[assignment]
            v_w = self.v_proj_weight  # type: ignore[assignment]

        if self.in_proj_bias is not None:
            bt = self.in_proj_bias._impl
            b_parts = _C_engine.split_at(bt, [d, 2 * d], 0)
            q_b: Tensor | None = _wrap(b_parts[0])
            k_b: Tensor | None = _wrap(b_parts[1])
            v_b: Tensor | None = _wrap(b_parts[2])
        else:
            q_b = k_b = v_b = None

        q: Tensor = linear(query, q_w, q_b)
        k: Tensor = linear(key, k_w, k_b)
        v: Tensor = linear(value, v_w, v_b)
        return q, k, v

    def _split_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Reshape ``(B, T, embed_dim)`` → ``(B, num_heads, T, head_dim)``."""
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            [0, 2, 1, 3]
        )

    def _merge_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """Inverse of :meth:`_split_heads`."""
        return x.permute([0, 2, 1, 3]).reshape(batch_size, seq_len, self.embed_dim)

    def _build_attn_mask(
        self,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        batch_size: int,
        target_len: int,
        source_len: int,
        float_dtype: object,
    ) -> Tensor | None:
        """Combine ``attn_mask`` and ``key_padding_mask`` into a single
        ``(B, H, T, S)`` additive float mask suitable for SDPA."""
        merged: Tensor | None = None

        if attn_mask is not None:
            am: Tensor = _to_additive_mask(attn_mask, float_dtype)
            if am.ndim == 2:
                am = am.reshape(1, 1, target_len, source_len)
            elif am.ndim == 3:
                # ``(B*H, T, S)`` is the reference convention for per-head masks.
                am = am.reshape(batch_size, self.num_heads, target_len, source_len)
            elif am.ndim == 4:
                pass
            else:
                raise ValueError(f"attn_mask must be 2-D, 3-D, or 4-D; got {am.ndim}-D")
            merged = am

        if key_padding_mask is not None:
            kpm: Tensor = _to_additive_mask(key_padding_mask, float_dtype)
            # (B, S) → (B, 1, 1, S) so it broadcasts over heads & target length.
            kpm = kpm.reshape(batch_size, 1, 1, source_len)
            merged = kpm if merged is None else merged + kpm

        return merged

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        # Internally operate in (B, T, E) layout regardless of batch_first.
        if not self.batch_first:
            query = query.permute([1, 0, 2])
            key = key.permute([1, 0, 2])
            value = value.permute([1, 0, 2])

        B: int = query.shape[0]
        Tq: int = query.shape[1]

        q, k, v = self._project_qkv(query, key, value)

        # ── add_bias_kv: prepend learnable bias rows to K / V ───────────
        if self.bias_k is not None:
            assert self.bias_v is not None
            bk: Tensor = self.bias_k.expand(B, 1, self.embed_dim)
            bv: Tensor = self.bias_v.expand(B, 1, self.embed_dim)
            k = _lucid.cat([k, bk], 1)
            v = _lucid.cat([v, bv], 1)

        # ── add_zero_attn: append a zero row to K / V ───────────────────
        if self.add_zero_attn:
            zero_kv: Tensor = _lucid.zeros(
                B, 1, self.embed_dim, dtype=k.dtype, device=k.device
            )
            k = _lucid.cat([k, zero_kv], 1)
            v = _lucid.cat([v, zero_kv], 1)

        Tk: int = k.shape[1]

        # Split heads.
        qh: Tensor = self._split_heads(q, B, Tq)
        kh: Tensor = self._split_heads(k, B, Tk)
        vh: Tensor = self._split_heads(v, B, Tk)

        merged_mask: Tensor | None = self._build_attn_mask(
            attn_mask, key_padding_mask, B, Tq, Tk, q.dtype
        )

        # Adding a bias / zero row to K means key_padding_mask (which is
        # only sized for the original sequence) needs to be extended.  We
        # simply do *not* extend it; those extra positions are always
        # attendable, matching the reference framework's behaviour.

        # Use the fused SDPA op only when there is no Python-built mask;
        # the engine's broadcasting rules for additive masks differ from
        # the manual softmax path and would silently desync.  Whenever
        # weights are needed, or a mask is involved, fall back to the
        # explicit Q·K^T → softmax path.
        attn_weights: Tensor | None
        if need_weights or merged_mask is not None:
            attn_out, _aw = self._attn_with_weights(qh, kh, vh, merged_mask, is_causal)
            attn_weights = _aw if need_weights else None
        else:
            attn_out = scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_weights = None

        # (B, H, Tq, head_dim) → (B, Tq, embed_dim)
        out: Tensor = self._merge_heads(attn_out, B, Tq)

        out = linear(out, self.out_proj_weight, self.out_proj_bias)

        if not self.batch_first:
            out = out.permute([1, 0, 2])

        if need_weights and attn_weights is not None and average_attn_weights:
            # (B, H, Tq, Tk) → (B, Tq, Tk)
            attn_weights = attn_weights.mean(dim=1)

        return out, attn_weights

    def _attn_with_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None,
        is_causal: bool,
    ) -> tuple[Tensor, Tensor]:
        """Manual scaled-dot-product attention that also returns weights.

        Used when the caller asks for attention weights — the fused SDPA
        op returns only the contracted output.  Performance impact is
        small for typical sequence lengths and only triggered on demand.
        """
        from lucid.nn.functional.activations import softmax as _softmax

        head_dim: int = q.shape[-1]
        scale: float = 1.0 / math.sqrt(head_dim)
        # scores: (B, H, Tq, Tk)
        scores: Tensor = _lucid.matmul(q, k.permute([0, 1, 3, 2])) * scale

        if is_causal:
            # Build an upper-triangular -inf mask.
            Tq: int = scores.shape[2]
            Tk: int = scores.shape[3]
            tri = _lucid.tensor(
                [[_NEG_INF if j > i else 0.0 for j in range(Tk)] for i in range(Tq)],
                dtype=scores.dtype,
                device=scores.device,
            )
            scores = scores + tri.reshape(1, 1, Tq, Tk)

        if attn_mask is not None:
            scores = scores + attn_mask

        weights: Tensor = _softmax(scores, dim=-1)
        if self.training and self.dropout > 0.0:
            from lucid.nn.functional.dropout import dropout as _dropout

            weights = _dropout(weights, p=self.dropout, training=True)
        out: Tensor = _lucid.matmul(weights, v)
        return out, weights

    def extra_repr(self) -> str:
        s: str = (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout}, batch_first={self.batch_first}"
        )
        if self.kdim != self.embed_dim:
            s += f", kdim={self.kdim}"
        if self.vdim != self.embed_dim:
            s += f", vdim={self.vdim}"
        if self.bias_k is not None:
            s += ", add_bias_kv=True"
        if self.add_zero_attn:
            s += ", add_zero_attn=True"
        return s
