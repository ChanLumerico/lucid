"""
nn.functional attention operations.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    r"""Scaled dot-product attention — the core of every Transformer block.

    Computes the attention-weighted aggregation of value vectors using
    query-key dot products as similarity scores:

    .. math::

        \mathrm{Attention}(Q, K, V) =
            \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V

    where :math:`M` is an optional additive mask (``-inf`` to disallow
    attention at a position, ``0`` to allow).  The :math:`1/\sqrt{d_k}`
    factor keeps the softmax in a usable temperature regime as the head
    dimension :math:`d_k` grows — without it, large dot products saturate
    the softmax into a near one-hot distribution and gradients vanish.

    Parameters
    ----------
    query : Tensor
        Shape ``(B, H, T, E)``.  ``B`` batch, ``H`` heads, ``T`` query
        positions, ``E`` head dimension :math:`d_k`.
    key : Tensor
        Shape ``(B, H, S, E)``.
    value : Tensor
        Shape ``(B, H, S, V)``.  ``V`` may differ from ``E``.
    attn_mask : Tensor, optional
        Additive mask broadcast-compatible with ``(B, H, T, S)``.  Use
        large negative values (or ``-inf``) at positions to mask out.
        Mutually exclusive with ``is_causal``.
    dropout_p : float, optional
        Dropout probability applied to attention weights during training.
        Default ``0.0``.
    is_causal : bool, optional
        If ``True``, apply an upper-triangular causal mask so each query
        position only attends to keys at the same or earlier positions
        (autoregressive decoder self-attention).
    scale : float, optional
        Override the default :math:`1/\sqrt{d_k}` scale factor.

    Returns
    -------
    Tensor
        Attention output of shape ``(B, H, T, V)``.

    Notes
    -----
    Introduced in *Attention Is All You Need* (Vaswani et al., 2017).
    The implementation uses the log-sum-exp form of softmax for numerical
    stability under aggressive masking, and fuses the scale into the
    score matrix prior to softmax.  Causal masking enables efficient
    autoregressive decoding when combined with a key/value cache.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import scaled_dot_product_attention
    >>> q = lucid.randn(2, 8, 16, 64)          # (B, H, T, E)
    >>> k = lucid.randn(2, 8, 16, 64)
    >>> v = lucid.randn(2, 8, 16, 64)
    >>> out = scaled_dot_product_attention(q, k, v, is_causal=True)
    >>> out.shape
    (2, 8, 16, 64)
    """
    import math

    mask = _unwrap(attn_mask) if attn_mask is not None else None
    head_dim = query.shape[-1]
    scale_val = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    return _wrap(
        _C_engine.nn.scaled_dot_product_attention(
            _unwrap(query),
            _unwrap(key),
            _unwrap(value),
            mask,
            scale_val,
            is_causal,
        )
    )
