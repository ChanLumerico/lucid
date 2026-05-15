"""Registry factories for BERT variants.

Sizes follow Devlin et al. (base / large) and Turc et al. 2019
("Well-Read Students Learn Better") for the four pre-distilled sizes
(tiny / mini / small / medium).  No pretrained weight URLs are registered yet
— follow-up PRs can wire HuggingFace ckpts through ``PretrainedEntry``.
"""

from lucid.models._registry import register_model
from lucid.models.text.bert._config import BertConfig
from lucid.models.text.bert._model import (
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
)

# Devlin et al. 2018 + Turc et al. 2019 size table.
_CFG_TINY = BertConfig(
    hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512
)
_CFG_MINI = BertConfig(
    hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024
)
_CFG_SMALL = BertConfig(
    hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048
)
_CFG_MEDIUM = BertConfig(
    hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048
)
_CFG_BASE = BertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
)
_CFG_LARGE = BertConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
)


def _apply(cfg: BertConfig, overrides: dict[str, object]) -> BertConfig:
    if not overrides:
        return cfg
    return BertConfig(**{**cfg.__dict__, **overrides})


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_TINY,
)
def bert_tiny(pretrained: bool = False, **overrides: object) -> BertModel:
    """BERT-Tiny (L=2, H=128) — Turc et al., 2019."""
    return BertModel(_apply(_CFG_TINY, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_MINI,
)
def bert_mini(pretrained: bool = False, **overrides: object) -> BertModel:
    return BertModel(_apply(_CFG_MINI, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_SMALL,
)
def bert_small(pretrained: bool = False, **overrides: object) -> BertModel:
    return BertModel(_apply(_CFG_SMALL, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_MEDIUM,
)
def bert_medium(pretrained: bool = False, **overrides: object) -> BertModel:
    return BertModel(_apply(_CFG_MEDIUM, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_BASE,
)
def bert_base(pretrained: bool = False, **overrides: object) -> BertModel:
    """BERT-Base (L=12, H=768) — Devlin et al., 2018."""
    return BertModel(_apply(_CFG_BASE, overrides))


@register_model(
    task="base",
    family="bert",
    model_type="bert",
    model_class=BertModel,
    default_config=_CFG_LARGE,
)
def bert_large(pretrained: bool = False, **overrides: object) -> BertModel:
    """BERT-Large (L=24, H=1024) — Devlin et al., 2018."""
    return BertModel(_apply(_CFG_LARGE, overrides))


# ── Masked-LM heads ───────────────────────────────────────────────────────────


@register_model(
    task="masked-lm",
    family="bert",
    model_type="bert",
    model_class=BertForMaskedLM,
    default_config=_CFG_BASE,
)
def bert_base_mlm(pretrained: bool = False, **overrides: object) -> BertForMaskedLM:
    return BertForMaskedLM(_apply(_CFG_BASE, overrides))


@register_model(
    task="masked-lm",
    family="bert",
    model_type="bert",
    model_class=BertForMaskedLM,
    default_config=_CFG_LARGE,
)
def bert_large_mlm(pretrained: bool = False, **overrides: object) -> BertForMaskedLM:
    return BertForMaskedLM(_apply(_CFG_LARGE, overrides))


# ── Sequence / token / QA classification heads ────────────────────────────────


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BertForSequenceClassification,
    default_config=_CFG_BASE,
)
def bert_base_cls(
    pretrained: bool = False, **overrides: object
) -> BertForSequenceClassification:
    return BertForSequenceClassification(_apply(_CFG_BASE, overrides))


@register_model(
    task="sequence-classification",
    family="bert",
    model_type="bert",
    model_class=BertForSequenceClassification,
    default_config=_CFG_LARGE,
)
def bert_large_cls(
    pretrained: bool = False, **overrides: object
) -> BertForSequenceClassification:
    return BertForSequenceClassification(_apply(_CFG_LARGE, overrides))


@register_model(
    task="token-classification",
    family="bert",
    model_type="bert",
    model_class=BertForTokenClassification,
    default_config=_CFG_BASE,
)
def bert_base_token_cls(
    pretrained: bool = False, **overrides: object
) -> BertForTokenClassification:
    return BertForTokenClassification(_apply(_CFG_BASE, overrides))


@register_model(
    task="question-answering",
    family="bert",
    model_type="bert",
    model_class=BertForQuestionAnswering,
    default_config=_CFG_BASE,
)
def bert_base_qa(
    pretrained: bool = False, **overrides: object
) -> BertForQuestionAnswering:
    return BertForQuestionAnswering(_apply(_CFG_BASE, overrides))
