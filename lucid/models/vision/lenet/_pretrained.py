"""Registry factories for LeNet (LeCun et al., 1998).

The original paper specifies a single architecture (tanh activations +
average pooling).  Modern ReLU / max-pool reimplementations are not
paper-defined variants — get them via ``create_model("lenet_5",
activation="relu", pooling="max")`` instead.
"""

from lucid.models._registry import register_model
from lucid.models.vision.lenet._config import LeNetConfig
from lucid.models.vision.lenet._model import LeNet, LeNetForImageClassification

_CFG_5 = LeNetConfig()  # paper original (tanh + avg-pool)


# ── Backbone ──────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="lenet",
    model_type="lenet",
    model_class=LeNet,
    default_config=_CFG_5,
)
def lenet_5(pretrained: bool = False, **overrides: object) -> LeNet:
    """LeNet-5 backbone (LeCun et al., 1998) — tanh + avg-pool."""
    cfg = LeNetConfig(**{**_CFG_5.__dict__, **overrides}) if overrides else _CFG_5
    return LeNet(cfg)


# ── Classifier ────────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="lenet",
    model_type="lenet",
    model_class=LeNetForImageClassification,
    default_config=_CFG_5,
)
def lenet_5_cls(
    pretrained: bool = False, **overrides: object
) -> LeNetForImageClassification:
    """LeNet-5 classifier (LeCun et al., 1998)."""
    cfg = LeNetConfig(**{**_CFG_5.__dict__, **overrides}) if overrides else _CFG_5
    return LeNetForImageClassification(cfg)
