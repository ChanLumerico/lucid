r"""
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


def _to_additive_mask(mask: "Tensor", float_dtype: object) -> "Tensor":
    """Convert a bool/byte mask (True = mask out) to an additive float mask
    (-inf where True, 0 where False).  Already-float masks pass through."""
    if mask.dtype == _lucid.bool_:
        # mask: True ⇒ -inf, False ⇒ 0
        _dtype = cast(DTypeLike, float_dtype)
        zero_t: Tensor = _lucid.zeros(mask.shape, dtype=_dtype, device=mask.device)
        ninf_t: Tensor = _lucid.full(
            mask.shape, _NEG_INF, dtype=_dtype, device=mask.device
        )
        return _lucid.where(mask, ninf_t, zero_t)
    return mask


class MultiheadAttention(Module):
    r"""Multi-head scaled dot-product attention.

    Implements the multi-head attention mechanism introduced in
    "Attention Is All You Need" (Vaswani et al., 2017).  Each head
    independently computes scaled dot-product attention over a
    learned linear projection of the query, key and value inputs;
    the per-head outputs are then concatenated and projected once
    more to produce the final result.

    **Scaled dot-product attention** for a single head:

    .. math::

        \text{Attention}(Q, K, V)
        = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V

    where :math:`d_k` is the per-head key dimension (``head_dim``).
    The :math:`1/\sqrt{d_k}` scaling prevents the dot-products from
    growing so large that the softmax function is pushed into regions
    with extremely small gradients.

    **Multi-head attention** uses :math:`h` parallel heads:

    .. math::

        \text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)

    .. math::

        \text{MultiHead}(Q, K, V)
        = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O

    where :math:`W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}`,
    :math:`W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}`,
    :math:`W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}`, and
    :math:`W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}` are
    learned projection matrices.

    When ``kdim`` and ``vdim`` both equal ``embed_dim``, the three
    input projections are stored as a single fused weight
    ``in_proj_weight`` of shape ``(3 * embed_dim, embed_dim)`` and
    split at runtime, which is more cache-friendly on Apple Silicon.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model, :math:`d_{\text{model}}`.
        Must be divisible by ``num_heads``.
    num_heads : int
        Number of parallel attention heads :math:`h`.
        Each head operates on a subspace of dimension
        ``head_dim = embed_dim // num_heads``.
    dropout : float, optional
        Dropout probability applied to the attention weight matrix
        during training.  Default: ``0.0``.
    bias : bool, optional
        If ``True``, learnable bias terms are added to all input and
        output projection layers.  Default: ``True``.
    add_bias_kv : bool, optional
        If ``True``, learnable bias rows ``bias_k`` and ``bias_v``
        (each of shape ``(1, 1, embed_dim)``) are appended to the
        key and value sequences along the sequence dimension before
        the attention computation.  Useful for cross-attention
        scenarios where extra context tokens are desired.
        Default: ``False``.
    add_zero_attn : bool, optional
        If ``True``, a zero-valued row is appended to the key and
        value sequences.  This can stabilise training in early steps
        by providing an "attend to nothing" option.  Default: ``False``.
    kdim : int or None, optional
        Feature dimension of the key input.  When ``None`` (default),
        falls back to ``embed_dim`` and a fused ``in_proj_weight``
        is used.
    vdim : int or None, optional
        Feature dimension of the value input.  When ``None``
        (default), falls back to ``embed_dim``.
    batch_first : bool, optional
        Controls the expected layout of all input and output tensors.

        * ``False`` (default): ``(seq_len, batch, embed_dim)``  — the
          classic sequence-first convention.
        * ``True``: ``(batch, seq_len, embed_dim)`` — more intuitive
          for most modern use-cases.

    device : DeviceLike, optional
        Device on which to allocate parameters.  ``None`` defaults to
        the current default device.
    dtype : DTypeLike, optional
        Data type for all parameters.  ``None`` defaults to the
        current default floating-point type.

    Attributes
    ----------
    embed_dim : int
        Total model dimension passed at construction.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Per-head dimension: ``embed_dim // num_heads``.
    kdim : int
        Effective key feature dimension.
    vdim : int
        Effective value feature dimension.
    dropout : float
        Attention weight dropout probability.
    batch_first : bool
        Whether inputs are ``(batch, seq, feature)``.
    in_proj_weight : Parameter or None
        Fused ``(3 * embed_dim, embed_dim)`` projection weight used
        when ``kdim == vdim == embed_dim``.  Sliced at runtime into
        Q, K, V sub-weights.  ``None`` when using separate weights.
    q_proj_weight : Parameter or None
        Separate query projection weight ``(embed_dim, embed_dim)``.
        Non-``None`` only when ``kdim`` or ``vdim`` differs from
        ``embed_dim``.
    k_proj_weight : Parameter or None
        Separate key projection weight ``(embed_dim, kdim)``.
    v_proj_weight : Parameter or None
        Separate value projection weight ``(embed_dim, vdim)``.
    in_proj_bias : Parameter or None
        Bias for the fused input projection ``(3 * embed_dim,)``.
        ``None`` when ``bias=False``.
    out_proj_weight : Parameter
        Output projection weight ``(embed_dim, embed_dim)``.
    out_proj_bias : Parameter or None
        Output projection bias ``(embed_dim,)``.
        ``None`` when ``bias=False``.
    bias_k : Parameter or None
        Learnable key bias row ``(1, 1, embed_dim)``.
        Non-``None`` when ``add_bias_kv=True``.
    bias_v : Parameter or None
        Learnable value bias row ``(1, 1, embed_dim)``.
        Non-``None`` when ``add_bias_kv=True``.
    add_zero_attn : bool
        Whether a zero row is appended to K and V.

    Shape
    -----
    The shapes below use the following notation:

    * :math:`N` — batch size
    * :math:`L` — target (query) sequence length
    * :math:`S` — source (key / value) sequence length
    * :math:`E` — ``embed_dim``

    When ``batch_first=False`` (default):

    * ``query``: :math:`(L, N, E)`
    * ``key``: :math:`(S, N, E_k)` where :math:`E_k` = ``kdim``
    * ``value``: :math:`(S, N, E_v)` where :math:`E_v` = ``vdim``
    * Output ``attn_output``: :math:`(L, N, E)`
    * Output ``attn_weights``: :math:`(N, L, S)` when
      ``need_weights=True`` and ``average_attn_weights=True``;
      :math:`(N, h, L, S)` when ``average_attn_weights=False``.

    When ``batch_first=True``:

    * ``query``: :math:`(N, L, E)`
    * ``key`` / ``value``: :math:`(N, S, E_{k/v})`
    * Output ``attn_output``: :math:`(N, L, E)`

    Notes
    -----
    **Why scale by** :math:`1/\sqrt{d_k}`?
        As :math:`d_k` grows, the dot-products :math:`QK^\top`
        accumulate over more dimensions and their magnitude grows
        like :math:`\sqrt{d_k}` under the assumption of unit-variance
        inputs.  Without the scale factor the softmax would saturate,
        producing near-one-hot distributions and vanishingly small
        gradients.  Dividing by :math:`\sqrt{d_k}` restores
        roughly unit variance before the softmax.

    **Causal masking** (``is_causal=True``):
        An upper-triangular :math:`-\infty` mask is added to the
        score matrix so that position :math:`i` cannot attend to any
        position :math:`j > i`.  This implements the autoregressive
        constraint needed for language model decoding.

    **Fused vs. separate projections**:
        When ``kdim == vdim == embed_dim``, the Q/K/V projections
        share a single ``(3E, E)`` weight matrix.  This layout
        allows a single ``linear`` call plus a cheap ``split_at``
        on the result, which amortises kernel-launch overhead and
        improves cache locality on the MLX / Accelerate backends.

    **Checkpoint compatibility**:
        State-dicts from the reference framework store the output
        projection under the key ``out_proj.weight`` / ``out_proj.bias``
        (a sub-module named ``out_proj``).  Lucid's ``_load_from_state_dict``
        hook transparently remaps those keys to the flat
        ``out_proj_weight`` / ``out_proj_bias`` attributes used here,
        so pre-trained weights can be loaded directly.

    Examples
    --------
    **Basic self-attention** (sequence-first layout):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)
    >>> # Sequence-first: (seq_len, batch, embed_dim)
    >>> x = lucid.randn(10, 2, 64)          # 10 tokens, batch=2
    >>> out, weights = mha(x, x, x)
    >>> out.shape
    (10, 2, 64)
    >>> weights.shape                        # averaged over heads
    (2, 10, 10)

    **Cross-attention with batch_first layout and causal mask**:

    >>> mha = nn.MultiheadAttention(embed_dim=64, num_heads=8,
    ...                             batch_first=True)
    >>> q = lucid.randn(2, 6, 64)           # batch=2, 6 query tokens
    >>> kv = lucid.randn(2, 10, 64)         # 10 key/value tokens
    >>> out, _ = mha(q, kv, kv, need_weights=False)
    >>> out.shape
    (2, 6, 64)

    **Cross-modal attention with different key/value dimensions**:

    >>> mha = nn.MultiheadAttention(embed_dim=128, num_heads=4,
    ...                             kdim=64, vdim=64)
    >>> q = lucid.randn(5, 1, 128)
    >>> kv = lucid.randn(7, 1, 64)
    >>> out, weights = mha(q, kv, kv)
    >>> out.shape
    (5, 1, 128)
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
        state_dict: dict[str, "Tensor"],
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
        self, query: "Tensor", key: "Tensor", value: "Tensor"
    ) -> tuple["Tensor", "Tensor", "Tensor"]:
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

    def _split_heads(self, x: "Tensor", batch_size: int, seq_len: int) -> "Tensor":
        """Reshape ``(B, T, embed_dim)`` → ``(B, num_heads, T, head_dim)``."""
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(
            [0, 2, 1, 3]
        )

    def _merge_heads(self, x: "Tensor", batch_size: int, seq_len: int) -> "Tensor":
        """Inverse of :meth:`_split_heads`."""
        return x.permute([0, 2, 1, 3]).reshape(batch_size, seq_len, self.embed_dim)

    def _build_attn_mask(
        self,
        attn_mask: "Tensor | None",
        key_padding_mask: "Tensor | None",
        batch_size: int,
        target_len: int,
        source_len: int,
        float_dtype: object,
    ) -> "Tensor | None":
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
        query: "Tensor",
        key: "Tensor",
        value: "Tensor",
        key_padding_mask: "Tensor | None" = None,
        need_weights: bool = True,
        attn_mask: "Tensor | None" = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple["Tensor", "Tensor | None"]:
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
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        attn_mask: "Tensor | None",
        is_causal: bool,
    ) -> tuple["Tensor", "Tensor"]:
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
