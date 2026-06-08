"""Registry factories for MobileNet v1."""

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.mobilenet._config import MobileNetV1Config
from lucid.models.vision.mobilenet._model import (
    MobileNetV1,
    MobileNetV1ForImageClassification,
)
from lucid.models.vision.mobilenet._weights import MobileNetV1Weights

_CFG_100 = MobileNetV1Config(width_mult=1.0)
_CFG_075 = MobileNetV1Config(width_mult=0.75)
_CFG_050 = MobileNetV1Config(width_mult=0.5)
_CFG_025 = MobileNetV1Config(width_mult=0.25)


def _b(cfg: MobileNetV1Config, kw: dict[str, object]) -> MobileNetV1:
    return MobileNetV1(replace(cfg, **cast(dict[str, Any], kw)) if kw else cfg)


def _c(
    cfg: MobileNetV1Config, kw: dict[str, object]
) -> MobileNetV1ForImageClassification:
    return MobileNetV1ForImageClassification(
        replace(cfg, **cast(dict[str, Any], kw)) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1,
    default_config=_CFG_100,
)
def mobilenet_v1(pretrained: bool = False, **overrides: object) -> MobileNetV1:
    r"""MobileNet-v1 backbone at width multiplier :math:`\alpha = 1.0`.

    Builds a :class:`MobileNetV1` with the canonical paper topology:
    a 3×3 stem (stride 2) followed by 13 depthwise+pointwise blocks,
    yielding approximately 4.2M parameters.  Howard et al., 2017
    report a 70.6% ImageNet-1k top-1 validation accuracy with this
    configuration (Table 7).  The default choice when the full
    accuracy budget is available.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV1Config`
        (e.g. ``in_channels=1`` for grayscale input).

    Returns
    -------
    MobileNetV1
        Backbone with the MobileNet-v1 (:math:`\alpha = 1.0`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 1 and Table 7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1
    >>> model = mobilenet_v1()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 1024, 1, 1)
    """
    return _b(_CFG_100, overrides)


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1,
    default_config=_CFG_075,
)
def mobilenet_v1_075(pretrained: bool = False, **overrides: object) -> MobileNetV1:
    r"""MobileNet-v1 backbone at width multiplier :math:`\alpha = 0.75`.

    Builds a :class:`MobileNetV1` with every channel count multiplied
    by 0.75 — approximately 2.6M parameters.  Howard et al., 2017
    report 68.4% ImageNet-1k top-1 accuracy with this configuration
    (Table 7), at roughly 60% of the FLOPs of the full-width model.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV1Config`.

    Returns
    -------
    MobileNetV1
        Backbone with the MobileNet-v1 (:math:`\alpha = 0.75`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1_075
    >>> model = mobilenet_v1_075()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 768, 1, 1)
    """
    return _b(_CFG_075, overrides)


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1,
    default_config=_CFG_050,
)
def mobilenet_v1_050(pretrained: bool = False, **overrides: object) -> MobileNetV1:
    r"""MobileNet-v1 backbone at width multiplier :math:`\alpha = 0.5`.

    Builds a :class:`MobileNetV1` with every channel count multiplied
    by 0.5 — approximately 1.3M parameters.  Howard et al., 2017
    report 63.7% ImageNet-1k top-1 accuracy with this configuration
    (Table 7), at roughly 27% of the FLOPs of the full-width model.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV1Config`.

    Returns
    -------
    MobileNetV1
        Backbone with the MobileNet-v1 (:math:`\alpha = 0.5`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1_050
    >>> model = mobilenet_v1_050()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 1, 1)
    """
    return _b(_CFG_050, overrides)


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1,
    default_config=_CFG_025,
)
def mobilenet_v1_025(pretrained: bool = False, **overrides: object) -> MobileNetV1:
    r"""MobileNet-v1 backbone at width multiplier :math:`\alpha = 0.25`.

    Builds a :class:`MobileNetV1` with every channel count multiplied
    by 0.25 — approximately 0.5M parameters.  Howard et al., 2017
    report 50.6% ImageNet-1k top-1 accuracy with this configuration
    (Table 7) — the smallest MobileNet-v1 variant, targeted at
    extreme edge deployments where parameter count is the binding
    constraint.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV1Config`.

    Returns
    -------
    MobileNetV1
        Backbone with the MobileNet-v1 (:math:`\alpha = 0.25`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1_025
    >>> model = mobilenet_v1_025()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 256, 1, 1)
    """
    return _b(_CFG_025, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


# reason: mobilenet_v1_cls adds typed weights= kwarg (per-model WeightsEnum);
# ModelFactory protocol predates the v3.1 weights system and still names only
# pretrained + **overrides.
@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1ForImageClassification,
    default_config=_CFG_100,
)
def mobilenet_v1_cls(
    pretrained: bool | str = False,
    *,
    weights: MobileNetV1Weights | None = None,
    **overrides: object,
) -> MobileNetV1ForImageClassification:
    r"""MobileNet-v1 image classifier at width multiplier :math:`\alpha = 1.0`.

    Builds a :class:`MobileNetV1ForImageClassification` with the
    canonical paper topology (13 depthwise+pointwise blocks) followed
    by global average pooling and a linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    Approximately 4.2M parameters and 70.6% ImageNet-1k top-1 in
    Howard et al., 2017 (Table 7).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`MobileNetV1Weights.RA4_E3600_R224_IN1K`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : MobileNetV1Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV1Config`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    MobileNetV1ForImageClassification
        Classifier with the MobileNet-v1 (:math:`\alpha = 1.0`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 7.  Pretrained weights are converted from timm's
    ``mobilenetv1_100.ra4_e3600_r224_in1k`` (75.4% top-1 at 224x224
    under the RA4 recipe) and hosted under ``lucid-dl/mobilenet-v1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1_cls
    >>> model = mobilenet_v1_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    entry = weights_mod.resolve_weights(MobileNetV1Weights, pretrained, weights)
    model = _c(_CFG_100, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mobilenet_v1_cls")
    return model


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1ForImageClassification,
    default_config=_CFG_075,
)
def mobilenet_v1_075_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV1ForImageClassification:
    r"""MobileNet-v1 image classifier at width multiplier :math:`\alpha = 0.75`.

    Builds a :class:`MobileNetV1ForImageClassification` with the
    paper-cited 0.75-width topology — approximately 2.6M parameters
    and 68.4% ImageNet-1k top-1 in Howard et al., 2017 (Table 7).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV1Config`.

    Returns
    -------
    MobileNetV1ForImageClassification
        Classifier with the MobileNet-v1 (:math:`\alpha = 0.75`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1_075_cls
    >>> model = mobilenet_v1_075_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _c(_CFG_075, overrides)


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1ForImageClassification,
    default_config=_CFG_050,
)
def mobilenet_v1_050_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV1ForImageClassification:
    r"""MobileNet-v1 image classifier at width multiplier :math:`\alpha = 0.5`.

    Builds a :class:`MobileNetV1ForImageClassification` with the
    paper-cited 0.5-width topology — approximately 1.3M parameters
    and 63.7% ImageNet-1k top-1 in Howard et al., 2017 (Table 7).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV1Config`.

    Returns
    -------
    MobileNetV1ForImageClassification
        Classifier with the MobileNet-v1 (:math:`\alpha = 0.5`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1_050_cls
    >>> model = mobilenet_v1_050_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _c(_CFG_050, overrides)


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1ForImageClassification,
    default_config=_CFG_025,
)
def mobilenet_v1_025_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV1ForImageClassification:
    r"""MobileNet-v1 image classifier at width multiplier :math:`\alpha = 0.25`.

    Builds a :class:`MobileNetV1ForImageClassification` with the
    paper-cited 0.25-width topology — approximately 0.5M parameters
    and 50.6% ImageNet-1k top-1 in Howard et al., 2017 (Table 7).
    The smallest variant in the family, targeted at extreme edge
    deployments.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV1Config`.

    Returns
    -------
    MobileNetV1ForImageClassification
        Classifier with the MobileNet-v1 (:math:`\alpha = 0.25`)
        configuration applied (or with ``overrides`` merged on top
        of it).

    Notes
    -----
    See Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017,
    Table 7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1_025_cls
    >>> model = mobilenet_v1_025_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _c(_CFG_025, overrides)
