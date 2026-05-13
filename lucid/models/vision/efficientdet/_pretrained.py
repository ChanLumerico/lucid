"""Registry factories for EfficientDet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.efficientdet._config import (
    EfficientDetConfig,
    efficientdet_config,
)
from lucid.models.vision.efficientdet._model import EfficientDetForObjectDetection


def _det(cfg: EfficientDetConfig, kw: dict[str, object]) -> EfficientDetForObjectDetection:
    return EfficientDetForObjectDetection(
        EfficientDetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=efficientdet_config(phi=0),
)
def efficientdet_d0(
    pretrained: bool = False,
    **overrides: object,
) -> EfficientDetForObjectDetection:
    """EfficientDet-D0 (φ=0): smallest compound-scaled detector.

    EfficientNet-B0 backbone, BiFPN (64ch, 3 repeats), head depth 3.
    """
    return _det(efficientdet_config(phi=0), overrides)


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=efficientdet_config(phi=1),
)
def efficientdet_d1(
    pretrained: bool = False,
    **overrides: object,
) -> EfficientDetForObjectDetection:
    """EfficientDet-D1 (φ=1): BiFPN 88ch, 4 repeats, head depth 3."""
    return _det(efficientdet_config(phi=1), overrides)


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=efficientdet_config(phi=2),
)
def efficientdet_d2(
    pretrained: bool = False,
    **overrides: object,
) -> EfficientDetForObjectDetection:
    """EfficientDet-D2 (φ=2): BiFPN 112ch, 5 repeats, head depth 3."""
    return _det(efficientdet_config(phi=2), overrides)


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=efficientdet_config(phi=3),
)
def efficientdet_d3(
    pretrained: bool = False,
    **overrides: object,
) -> EfficientDetForObjectDetection:
    """EfficientDet-D3 (φ=3): BiFPN 160ch, 6 repeats, head depth 4."""
    return _det(efficientdet_config(phi=3), overrides)


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=efficientdet_config(phi=4),
)
def efficientdet_d4(
    pretrained: bool = False,
    **overrides: object,
) -> EfficientDetForObjectDetection:
    """EfficientDet-D4 (φ=4): BiFPN 224ch, 7 repeats, head depth 4."""
    return _det(efficientdet_config(phi=4), overrides)


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=efficientdet_config(phi=5),
)
def efficientdet_d5(
    pretrained: bool = False,
    **overrides: object,
) -> EfficientDetForObjectDetection:
    """EfficientDet-D5 (φ=5): BiFPN 288ch, 7 repeats, head depth 4."""
    return _det(efficientdet_config(phi=5), overrides)


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=efficientdet_config(phi=6),
)
def efficientdet_d6(
    pretrained: bool = False,
    **overrides: object,
) -> EfficientDetForObjectDetection:
    """EfficientDet-D6 (φ=6): BiFPN 384ch, 8 repeats, head depth 5."""
    return _det(efficientdet_config(phi=6), overrides)


@register_model(
    task="object-detection",
    family="efficientdet",
    model_type="efficientdet",
    model_class=EfficientDetForObjectDetection,
    default_config=efficientdet_config(phi=7),
)
def efficientdet_d7(
    pretrained: bool = False,
    **overrides: object,
) -> EfficientDetForObjectDetection:
    """EfficientDet-D7 (φ=7): largest variant, BiFPN 384ch, 8 repeats, head depth 5."""
    return _det(efficientdet_config(phi=7), overrides)
