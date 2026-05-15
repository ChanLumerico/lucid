"""Registry factories for GPT-2 variants (Radford et al., 2019).

Canonical sizes from the OpenAI release.  No pretrained weight URLs are
wired in yet — follow-up PR.
"""

from lucid.models._registry import register_model
from lucid.models.text.gpt2._config import GPT2Config
from lucid.models.text.gpt2._model import (
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Model,
)

# OpenAI release table — Radford et al., 2019 §2.3 (the only published
# GPT-2 sizes are small / medium / large / XL).
_CFG_SMALL = GPT2Config()  # 124M — the default
_CFG_MEDIUM = GPT2Config(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
)
_CFG_LARGE = GPT2Config(
    hidden_size=1280,
    num_hidden_layers=36,
    num_attention_heads=20,
    intermediate_size=5120,
)
_CFG_XLARGE = GPT2Config(
    hidden_size=1600,
    num_hidden_layers=48,
    num_attention_heads=25,
    intermediate_size=6400,
)


def _apply(cfg: GPT2Config, overrides: dict[str, object]) -> GPT2Config:
    return GPT2Config(**{**cfg.__dict__, **overrides}) if overrides else cfg


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2Model,
    default_config=_CFG_SMALL,
)
def gpt2_small(pretrained: bool = False, **overrides: object) -> GPT2Model:
    """GPT-2 small (124M) — Radford et al., 2019."""
    return GPT2Model(_apply(_CFG_SMALL, overrides))


@register_model(
    task="base",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2Model,
    default_config=_CFG_MEDIUM,
)
def gpt2_medium(pretrained: bool = False, **overrides: object) -> GPT2Model:
    """GPT-2 medium (355M)."""
    return GPT2Model(_apply(_CFG_MEDIUM, overrides))


@register_model(
    task="base",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2Model,
    default_config=_CFG_LARGE,
)
def gpt2_large(pretrained: bool = False, **overrides: object) -> GPT2Model:
    """GPT-2 large (774M)."""
    return GPT2Model(_apply(_CFG_LARGE, overrides))


@register_model(
    task="base",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2Model,
    default_config=_CFG_XLARGE,
)
def gpt2_xlarge(pretrained: bool = False, **overrides: object) -> GPT2Model:
    """GPT-2 XL (1.5B)."""
    return GPT2Model(_apply(_CFG_XLARGE, overrides))


# ── Causal-LM heads ───────────────────────────────────────────────────────────


@register_model(
    task="causal-lm",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2LMHeadModel,
    default_config=_CFG_SMALL,
)
def gpt2_small_lm(pretrained: bool = False, **overrides: object) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(_apply(_CFG_SMALL, overrides))


@register_model(
    task="causal-lm",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2LMHeadModel,
    default_config=_CFG_MEDIUM,
)
def gpt2_medium_lm(pretrained: bool = False, **overrides: object) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(_apply(_CFG_MEDIUM, overrides))


@register_model(
    task="causal-lm",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2LMHeadModel,
    default_config=_CFG_LARGE,
)
def gpt2_large_lm(pretrained: bool = False, **overrides: object) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(_apply(_CFG_LARGE, overrides))


@register_model(
    task="causal-lm",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2LMHeadModel,
    default_config=_CFG_XLARGE,
)
def gpt2_xlarge_lm(pretrained: bool = False, **overrides: object) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(_apply(_CFG_XLARGE, overrides))


# ── Sequence-classification head ──────────────────────────────────────────────


@register_model(
    task="sequence-classification",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2ForSequenceClassification,
    default_config=_CFG_SMALL,
)
def gpt2_small_cls(
    pretrained: bool = False, **overrides: object
) -> GPT2ForSequenceClassification:
    return GPT2ForSequenceClassification(_apply(_CFG_SMALL, overrides))
