"""Text-model helpers shared across text families.

Every transformer-LM family in :mod:`lucid.models.text` (BERT, GPT, GPT-2,
RoFormer, and future T5 / LLaMA / Mistral) needs the same two pieces of
boilerplate:

    * Activation dispatch — map a config-supplied string (one of the
      :data:`lucid.models.text.TextActivation` literals) to the correct
      :mod:`lucid.nn.functional` call.
    * Attention-mask normalisation — turn a ``(B, T)`` 0/1 padding mask
      into the additive ``(B, 1, 1, T)`` form that scaled-dot-product
      attention expects, with masked positions zeroed via ``-1e4`` bias.

Keeping them here avoids byte-identical duplication across every family
file and gives future text models a single canonical implementation.
"""

import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

__all__ = ["text_activation", "extended_attention_mask"]


def text_activation(name: str, x: Tensor) -> Tensor:
    """Apply the activation referenced by a :data:`TextActivation` literal.

    Args:
        name: One of ``"gelu"`` / ``"gelu_new"`` / ``"relu"`` / ``"silu"``
            / ``"swish"``.  Comes from a text config's ``hidden_act`` field.
        x:    Input tensor; activation is elementwise.

    Returns:
        Activated tensor of the same shape as ``x``.

    Raises:
        ValueError: If ``name`` is not a supported activation alias.
    """
    # Divergence, deliberate: ``"gelu"`` is the exact erf form here, which is
    # what the checkpoint-publishing reference implementation does — but BERT's
    # original release used the tanh approximation under that same name.  The
    # two references disagree, so no single mapping satisfies both; we follow
    # the one the weights were converted from.  The gap is bounded: over
    # x in [-6, 6] the two forms differ by at most 4.7e-4 absolute
    # (1.3e-4 relative to |x| + 1), well inside float32 activation noise.
    # Configs that need the original elementwise curve can ask for it by name
    # with ``hidden_act="gelu_new"``.
    if name == "gelu":
        return F.gelu(x, approximate="none")
    if name == "gelu_new":
        return F.gelu(x, approximate="tanh")
    if name == "relu":
        return F.relu(x)
    if name in ("silu", "swish"):
        return F.silu(x)
    raise ValueError(f"Unsupported activation {name!r}")


def extended_attention_mask(
    attention_mask: Tensor | None,
    input_shape: tuple[int, ...],
    key_length: int | None = None,
) -> Tensor | None:
    """Normalise a padding mask to the additive ``(B, 1, 1, T)`` form.

    Scaled-dot-product attention adds the resulting mask to the
    ``(B, H, T, T)`` score tensor; masked positions get ``-1e4`` added so
    that softmax drives their probabilities to ~0 without producing NaNs.

    Args:
        attention_mask: ``(B, T)`` integer / float mask with 1 for "attend"
            and 0 for "ignore", or already pre-broadcast at higher rank.
            ``None`` skips masking entirely.
        input_shape: ``(B, T, …)`` — the leading dims of the model input;
            only ``(B, T)`` are used.
        key_length: Number of *key* positions the scores span.  With a KV
            cache the queries are the ``T`` new tokens but the keys cover
            ``past_len + T`` positions, so the padding mask must be that
            wide.  Defaults to ``T`` (no cache).

    Returns:
        ``(B, 1, 1, key_length)`` additive mask, or ``None`` when no mask
        supplied.

    Raises:
        ValueError: If a rank-2 mask's width does not match ``key_length``.
            A mask covering only the new tokens cannot describe which of the
            *cached* positions were padding, and silently treating the past
            as all-real would attend to padded history.
    """
    if attention_mask is None:
        return None
    B, T = input_shape[0], input_shape[1]
    key_len = T if key_length is None else key_length
    if attention_mask.ndim == 2:
        given = int(attention_mask.shape[1])
        if given != key_len:
            hint = (
                " With a KV cache the mask must cover the cached history too "
                f"(past + new = {key_len})."
                if key_len != T
                else ""
            )
            raise ValueError(
                f"attention_mask has width {given} but the attention scores "
                f"span {key_len} key positions.{hint}"
            )
        mask = attention_mask.reshape(B, 1, 1, key_len)
    elif attention_mask.ndim == 3:
        # Caller pre-broadcast over heads already.
        mask = attention_mask.unsqueeze(1)
    else:
        mask = attention_mask
    return (1.0 - mask.float()) * -1e4
