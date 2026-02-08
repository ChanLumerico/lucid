from lucid.nn._kernel.attention import scaled_dot_product_attention_kernel

from lucid._tensor import Tensor


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    output_weight: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    op = scaled_dot_product_attention_kernel(
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        dropout_p=dropout_p,
    )
    attn_output = op(query, key, value)
    if not output_weight:
        return attn_output
    return attn_output, op.get_attention_weight(device=query.device)
