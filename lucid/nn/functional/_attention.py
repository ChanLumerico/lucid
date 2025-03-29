import math
import lucid
import lucid.nn.functional as F

from lucid._tensor import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = lucid.zeros(L, S, dtype=query.dtype).free()

    if is_causal:
        assert attn_mask is None
        temp_mask = 1 - lucid.ones(L, S).tril()
        attn_bias += temp_mask * -1e12

    if attn_mask is not None:
        attn_bias += attn_mask

    attn_weight = query @ key.mT * scale_factor
    attn_weight += attn_bias.broadcast_to(attn_weight.shape)
    attn_weight = F.softmax(attn_weight, axis=-1)
    attn_weight = F.dropout(attn_weight, dropout_p)

    return attn_weight @ value
