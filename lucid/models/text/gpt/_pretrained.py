"""Registry factories for GPT-1 (Radford et al., 2018).

The original paper specifies a single architecture: ``L=12, H=768, A=12``.
That's the only published variant; we therefore expose just one nominal
factory ``gpt`` (plus the ``_lm`` and ``_cls`` task heads).  Test code
that needs a small variant should override config fields via
``create_model("gpt", hidden_size=..., num_hidden_layers=..., ...)``.
"""

from lucid.models._registry import register_model
from lucid.models.text.gpt._config import GPTConfig
from lucid.models.text.gpt._model import (
    GPTForSequenceClassification,
    GPTLMHeadModel,
    GPTModel,
)

_CFG_BASE = GPTConfig()  # Radford et al. defaults — the paper's only size


def _apply(cfg: GPTConfig, overrides: dict[str, object]) -> GPTConfig:
    return GPTConfig(**{**cfg.__dict__, **overrides}) if overrides else cfg


# ── Backbone ──────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="gpt",
    model_type="gpt",
    model_class=GPTModel,
    default_config=_CFG_BASE,
)
def gpt(pretrained: bool = False, **overrides: object) -> GPTModel:
    """GPT-1 (Radford et al., 2018) — L=12, H=768, A=12."""
    return GPTModel(_apply(_CFG_BASE, overrides))


# ── Causal-LM head (GenerationMixin host) ─────────────────────────────────────


@register_model(
    task="causal-lm",
    family="gpt",
    model_type="gpt",
    model_class=GPTLMHeadModel,
    default_config=_CFG_BASE,
)
def gpt_lm(pretrained: bool = False, **overrides: object) -> GPTLMHeadModel:
    return GPTLMHeadModel(_apply(_CFG_BASE, overrides))


# ── Sequence-classification head ──────────────────────────────────────────────


@register_model(
    task="sequence-classification",
    family="gpt",
    model_type="gpt",
    model_class=GPTForSequenceClassification,
    default_config=_CFG_BASE,
)
def gpt_cls(
    pretrained: bool = False, **overrides: object
) -> GPTForSequenceClassification:
    return GPTForSequenceClassification(_apply(_CFG_BASE, overrides))
