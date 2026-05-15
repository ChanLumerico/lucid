"""Registry factories for Inception-ResNet v2."""

from lucid.models._registry import register_model
from lucid.models.vision.inception_resnet._config import InceptionResNetConfig
from lucid.models.vision.inception_resnet._model import (
    InceptionResNetV2,
    InceptionResNetV2ForImageClassification,
)

_CFG = InceptionResNetConfig()


@register_model(
    task="base",
    family="inception_resnet",
    model_type="inception_resnet",
    model_class=InceptionResNetV2,
    default_config=_CFG,
)
def inception_resnet_v2(
    pretrained: bool = False, **overrides: object
) -> InceptionResNetV2:
    """Inception-ResNet v2 backbone (Szegedy et al., 2016)."""
    cfg = InceptionResNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return InceptionResNetV2(cfg)


@register_model(
    task="image-classification",
    family="inception_resnet",
    model_type="inception_resnet",
    model_class=InceptionResNetV2ForImageClassification,
    default_config=_CFG,
)
def inception_resnet_v2_cls(
    pretrained: bool = False, **overrides: object
) -> InceptionResNetV2ForImageClassification:
    """Inception-ResNet v2 classifier (Szegedy et al., 2016)."""
    cfg = InceptionResNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return InceptionResNetV2ForImageClassification(cfg)
