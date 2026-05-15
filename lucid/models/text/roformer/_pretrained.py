"""Registry factories for RoFormer (Su et al., 2021).

The paper specifies a single architecture (``L=12, H=768, A=12`` — same
shape as BERT-base, with absolute PE replaced by RoPE).  We therefore
expose just ``roformer`` (plus task heads).  Test code that needs a
small variant should override config fields via
``create_model("roformer", hidden_size=..., num_hidden_layers=..., ...)``.
"""

from lucid.models._registry import register_model
from lucid.models.text.roformer._config import RoFormerConfig
from lucid.models.text.roformer._model import (
    RoFormerForMaskedLM,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerModel,
)

_CFG_BASE = RoFormerConfig()  # paper default — the only published size


def _apply(cfg: RoFormerConfig, overrides: dict[str, object]) -> RoFormerConfig:
    return RoFormerConfig(**{**cfg.__dict__, **overrides}) if overrides else cfg


# ── Backbone ──────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerModel,
    default_config=_CFG_BASE,
)
def roformer(pretrained: bool = False, **overrides: object) -> RoFormerModel:
    """RoFormer-Base (L=12, H=768) — Su et al., 2021."""
    return RoFormerModel(_apply(_CFG_BASE, overrides))


# ── Task heads ────────────────────────────────────────────────────────────────


@register_model(
    task="masked-lm",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForMaskedLM,
    default_config=_CFG_BASE,
)
def roformer_mlm(pretrained: bool = False, **overrides: object) -> RoFormerForMaskedLM:
    return RoFormerForMaskedLM(_apply(_CFG_BASE, overrides))


@register_model(
    task="sequence-classification",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForSequenceClassification,
    default_config=_CFG_BASE,
)
def roformer_cls(
    pretrained: bool = False, **overrides: object
) -> RoFormerForSequenceClassification:
    return RoFormerForSequenceClassification(_apply(_CFG_BASE, overrides))


@register_model(
    task="token-classification",
    family="roformer",
    model_type="roformer",
    model_class=RoFormerForTokenClassification,
    default_config=_CFG_BASE,
)
def roformer_token_cls(
    pretrained: bool = False, **overrides: object
) -> RoFormerForTokenClassification:
    return RoFormerForTokenClassification(_apply(_CFG_BASE, overrides))
