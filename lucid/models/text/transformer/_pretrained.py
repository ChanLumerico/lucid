"""Registry factories for the original Vaswani Transformer."""

from lucid.models._registry import register_model
from lucid.models.text.transformer._config import TransformerConfig
from lucid.models.text.transformer._model import (
    TransformerForSeq2SeqLM,
    TransformerForSequenceClassification,
    TransformerForTokenClassification,
    TransformerModel,
)

# Vaswani et al. (2017) §6.2 — Table 3 "base" and "big" (the only two
# sizes the paper specifies).
_CFG_BASE = TransformerConfig()
_CFG_LARGE = TransformerConfig(
    hidden_size=1024,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_dropout=0.3,
    attention_dropout=0.3,
)


def _apply(cfg: TransformerConfig, overrides: dict[str, object]) -> TransformerConfig:
    return TransformerConfig(**{**cfg.__dict__, **overrides}) if overrides else cfg


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="transformer",
    model_type="transformer",
    model_class=TransformerModel,
    default_config=_CFG_BASE,
)
def transformer_base(pretrained: bool = False, **overrides: object) -> TransformerModel:
    """Vaswani et al. (2017) "base" — L=6, d_model=512, h=8."""
    return TransformerModel(_apply(_CFG_BASE, overrides))


@register_model(
    task="base",
    family="transformer",
    model_type="transformer",
    model_class=TransformerModel,
    default_config=_CFG_LARGE,
)
def transformer_large(
    pretrained: bool = False, **overrides: object
) -> TransformerModel:
    """Vaswani et al. (2017) "big" — L=6, d_model=1024, h=16."""
    return TransformerModel(_apply(_CFG_LARGE, overrides))


# ── Seq2SeqLM heads ───────────────────────────────────────────────────────────


@register_model(
    task="seq2seq-lm",
    family="transformer",
    model_type="transformer",
    model_class=TransformerForSeq2SeqLM,
    default_config=_CFG_BASE,
)
def transformer_base_seq2seq(
    pretrained: bool = False, **overrides: object
) -> TransformerForSeq2SeqLM:
    return TransformerForSeq2SeqLM(_apply(_CFG_BASE, overrides))


@register_model(
    task="seq2seq-lm",
    family="transformer",
    model_type="transformer",
    model_class=TransformerForSeq2SeqLM,
    default_config=_CFG_LARGE,
)
def transformer_large_seq2seq(
    pretrained: bool = False, **overrides: object
) -> TransformerForSeq2SeqLM:
    return TransformerForSeq2SeqLM(_apply(_CFG_LARGE, overrides))


# ── Encoder-only downstream heads ─────────────────────────────────────────────


@register_model(
    task="sequence-classification",
    family="transformer",
    model_type="transformer",
    model_class=TransformerForSequenceClassification,
    default_config=_CFG_BASE,
)
def transformer_base_cls(
    pretrained: bool = False, **overrides: object
) -> TransformerForSequenceClassification:
    return TransformerForSequenceClassification(_apply(_CFG_BASE, overrides))


@register_model(
    task="token-classification",
    family="transformer",
    model_type="transformer",
    model_class=TransformerForTokenClassification,
    default_config=_CFG_BASE,
)
def transformer_base_token_cls(
    pretrained: bool = False, **overrides: object
) -> TransformerForTokenClassification:
    return TransformerForTokenClassification(_apply(_CFG_BASE, overrides))
