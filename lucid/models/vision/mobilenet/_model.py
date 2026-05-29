"""MobileNet v1 backbone and classifier (Howard et al., 2017).

Paper: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"

Key idea: replace standard Conv2d with *depthwise separable convolutions* —
  Depthwise  : one filter per input channel (groups=in_channels)
  Pointwise  : 1×1 Conv to project to the desired output channels

This reduces computation from M·N·Dk²·Df² to M·Dk²·Df² + M·N·Df²,
roughly a 8–9× reduction for 3×3 convolutions.

Architecture (width_mult=1.0):
  Conv   : 3→32, 3×3, s2
  DW+PW  : 32→64, s1
  DW+PW  : 64→128, s2
  DW+PW  : 128→128, s1
  DW+PW  : 128→256, s2
  DW+PW  : 256→256, s1
  DW+PW  : 256→512, s2
  DW+PW × 5 : 512→512, s1
  DW+PW  : 512→1024, s2
  DW+PW  : 1024→1024, s1
  AvgPool(7×7) → FC(1024, num_classes)
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import make_divisible as _make_divisible
from lucid.models.vision.mobilenet._config import MobileNetV1Config


def _dw_pw(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    """Depthwise + pointwise block with BN and ReLU6."""
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU6(inplace=True),
        # Pointwise
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


# (out_channels, stride) specs for the 13 DW+PW layers (at width_mult=1.0)
_DW_PW_SPECS: list[tuple[int, int]] = [
    (64, 1),
    (128, 2),
    (128, 1),
    (256, 2),
    (256, 1),
    (512, 2),
    (512, 1),
    (512, 1),
    (512, 1),
    (512, 1),
    (512, 1),
    (1024, 2),
    (1024, 1),
]


def _build_features(cfg: MobileNetV1Config) -> tuple[nn.Sequential, int]:
    """Build the full feature extractor. Returns (Sequential, num_out_channels)."""
    w = cfg.width_mult

    def _ch(c: int) -> int:
        return _make_divisible(c * w)

    first_ch = _ch(32)
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, first_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(first_ch),
        nn.ReLU6(inplace=True),
    ]
    in_ch = first_ch
    for out_c, stride in _DW_PW_SPECS:
        out_ch = _ch(out_c)
        layers.append(_dw_pw(in_ch, out_ch, stride))
        in_ch = out_ch

    return nn.Sequential(*layers), in_ch


# ---------------------------------------------------------------------------
# MobileNet v1 backbone  (task="base")
# ---------------------------------------------------------------------------


class MobileNetV1(PretrainedModel, BackboneMixin):
    r"""MobileNet v1 feature-extracting backbone (no classification head).

    Implements the depthwise-separable convolutional topology from
    Howard et al., "MobileNets: Efficient Convolutional Neural
    Networks for Mobile Vision Applications", arXiv:1704.04861, 2017.
    The network alternates a per-channel :math:`3 \times 3` depthwise
    convolution with a :math:`1 \times 1` pointwise convolution that
    mixes channels — a factorisation that reduces FLOPs by roughly
    :math:`1/N + 1/D_K^2` relative to a standard
    :math:`D_K \times D_K` convolution, an 8–9× saving for the
    :math:`3 \times 3` case.

    The body is a 3×3 stem (stride 2) followed by 13 depthwise +
    pointwise blocks producing feature maps at strides 2, 4, 8, 16,
    and 32 relative to the input.  A global average pool collapses
    the final spatial map to a single 1×1 descriptor.  This class
    is the canonical feature extractor used by detection /
    segmentation heads when deployed on mobile / embedded devices.

    Parameters
    ----------
    config : MobileNetV1Config
        Frozen architecture spec.  Use the factory functions
        (:func:`mobilenet_v1`, :func:`mobilenet_v1_075`, …) for
        paper-cited width-multiplier variants.

    Attributes
    ----------
    config : MobileNetV1Config
        Stored copy of the config that built this model.
    features : nn.Sequential
        Stem :math:`3 \times 3` conv (stride 2) followed by 13
        depthwise+pointwise blocks; channel counts scale with
        ``config.width_mult``.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        ``(B, C, 1, 1)``.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed
        via :class:`BackboneMixin` for downstream FPN / decoder
        modules.

    Notes
    -----
    Each depthwise-separable block computes

    .. math::

        \mathrm{PW}\big(\sigma(\mathrm{BN}(\mathrm{DW}_{3\times3}(x)))\big)
        \;\;\to\;\; \sigma(\mathrm{BN}(\,\cdot\,)),

    where :math:`\mathrm{DW}_{3\times3}` is the per-channel
    :math:`3 \times 3` spatial filter (``groups=in_channels``),
    :math:`\mathrm{PW}` is the :math:`1 \times 1` channel-mixing
    convolution, and :math:`\sigma` is ReLU.  The width multiplier
    :math:`\alpha \in (0, 1]` uniformly scales every channel count
    so that the same architecture can target sub-mW edge devices
    (:math:`\alpha = 0.25`) up to full desktop deployment
    (:math:`\alpha = 1.0`).

    Examples
    --------
    Build a MobileNet-v1 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1
    >>> backbone = mobilenet_v1()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape
    (2, 1024, 1, 1)

    Inspect the per-stage feature map descriptors for FPN integration:

    >>> info = backbone.feature_info
    >>> [(fi.stage, fi.num_channels, fi.reduction) for fi in info]
    [(1, 64, 2), (2, 128, 4), (3, 256, 8), (4, 512, 16), (5, 1024, 32)]
    """

    config_class: ClassVar[type[MobileNetV1Config]] = MobileNetV1Config
    base_model_prefix: ClassVar[str] = "mobilenet_v1"

    def __init__(self, config: MobileNetV1Config) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        w = config.width_mult

        def _ch(c: int) -> int:
            return _make_divisible(c * w)

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=_ch(64), reduction=2),
            FeatureInfo(stage=2, num_channels=_ch(128), reduction=4),
            FeatureInfo(stage=3, num_channels=_ch(256), reduction=8),
            FeatureInfo(stage=4, num_channels=_ch(512), reduction=16),
            FeatureInfo(stage=5, num_channels=_ch(1024), reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.features(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# MobileNet v1 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MobileNetV1ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    r"""MobileNet v1 with global-average-pooled linear classification head.

    Combines a :class:`MobileNetV1` backbone with the standard
    ImageNet classification head: an :class:`~lucid.nn.AdaptiveAvgPool2d`
    to pool every spatial location into a single feature vector,
    a :class:`~lucid.nn.Dropout` (probability ``config.dropout``,
    default 0.001 per the paper), and a :class:`~lucid.nn.Linear`
    projection to ``config.num_classes`` logits.  When ``labels``
    are supplied to :meth:`forward`, a cross-entropy loss is
    computed and returned alongside the logits.

    Parameters
    ----------
    config : MobileNetV1Config
        Architecture spec.  Use the ``*_cls`` factory functions
        (:func:`mobilenet_v1_cls`, :func:`mobilenet_v1_075_cls`, …)
        to obtain a paper-cited configuration; pass a custom config
        to retarget ``num_classes`` or change ``width_mult``.

    Attributes
    ----------
    config : MobileNetV1Config
        Stored copy of the config that built this model.
    features : nn.Sequential
        Same depthwise-separable stack as on :class:`MobileNetV1`.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool collapsing the final feature map to
        ``1 × 1``.
    drop : nn.Dropout
        Dropout layer (probability ``config.dropout``) applied
        before the linear classifier.
    classifier : nn.Linear
        Linear projection from ``round(1024 * width_mult)`` to
        ``config.num_classes``.

    Notes
    -----
    The classification flow is

    .. math::

        \text{logits} = W \,\operatorname{Drop}\!\left(
            \operatorname{GAP}(\,\mathrm{backbone}(x)\,)
        \right) + b,

    where :math:`\operatorname{GAP}` is the global average pool.
    Loss is the standard categorical cross-entropy, computed only
    when ``labels`` is not ``None``.

    Examples
    --------
    Run inference on a batch of 224×224 RGB images:

    >>> import lucid
    >>> from lucid.models.vision.mobilenet import mobilenet_v1_cls
    >>> model = mobilenet_v1_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)
    >>> out.loss is None
    True

    Compute a loss given labels (e.g. during training):

    >>> labels = lucid.tensor([0, 1, 2, 3], dtype=lucid.int64)
    >>> out = model(x, labels=labels)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[MobileNetV1Config]] = MobileNetV1Config
    base_model_prefix: ClassVar[str] = "mobilenet_v1"

    def __init__(self, config: MobileNetV1Config) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=config.dropout)
        self._build_classifier(num_features, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.features(x))
        x = cast(Tensor, self.avgpool(x))
        x = cast(Tensor, self.drop(x.flatten(1)))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
