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

    Returns:
        ``(B, 1, 1, T)`` additive mask, or ``None`` when no mask supplied.
    """
    if attention_mask is None:
        return None
    B, T = input_shape[0], input_shape[1]
    if attention_mask.ndim == 2:
        mask = attention_mask.reshape(B, 1, 1, T)
    elif attention_mask.ndim == 3:
        # Caller pre-broadcast over heads already.
        mask = attention_mask.unsqueeze(1)
    else:
        mask = attention_mask
    return (1.0 - mask.float()) * -1e4
