"""ZFNet backbone and classification head (Zeiler & Fergus, 2013).

Paper: "Visualizing and Understanding Convolutional Networks"
Architecture differs from AlexNet in the first two conv layers:
    Conv1 : 3→96,  7×7, stride=2, pad=1  → ReLU → LRN → MaxPool 3×3 s2
    Conv2 : 96→256, 5×5, stride=2, pad=0 → ReLU → LRN → MaxPool 3×3 s2
    Conv3 : 256→384, 3×3, pad=1          → ReLU
    Conv4 : 384→384, 3×3, pad=1          → ReLU
    Conv5 : 384→256, 3×3, pad=1          → ReLU → MaxPool 3×3 s2
    AdaptiveAvgPool → 6×6
    FC6   : 256*6*6 → 4096               → ReLU → Dropout
    FC7   : 4096    → 4096               → ReLU → Dropout
    FC8   : 4096    → num_classes
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.zfnet._config import ZFNetConfig


def _build_features(cfg: ZFNetConfig) -> nn.Sequential:
    return nn.Sequential(
        # Block 1 — 7×7 stride=2 (key ZFNet modification)
        nn.Conv2d(cfg.in_channels, 96, 7, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
        nn.MaxPool2d(3, stride=2),
        # Block 2 — 5×5 stride=2 (key ZFNet modification)
        nn.Conv2d(96, 256, 5, stride=2),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
        nn.MaxPool2d(3, stride=2),
        # Block 3
        nn.Conv2d(256, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # Block 4
        nn.Conv2d(384, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # Block 5
        nn.Conv2d(384, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2),
    )


# ---------------------------------------------------------------------------
# ZFNet backbone  (task="base")
# ---------------------------------------------------------------------------


class ZFNet(PretrainedModel, BackboneMixin):
    r"""ZFNet feature-extracting backbone (no fully-connected head).

    Implements the five-stage convolutional trunk from Zeiler & Fergus,
    "Visualizing and Understanding Convolutional Networks", ECCV 2014.
    ZFNet is best understood as an AlexNet that was *retuned by
    inspecting what each filter had learned*: the deconvolutional-
    network visualisation technique introduced in the same paper
    revealed that AlexNet's :math:`11\times11` stride-4 first
    convolution was producing a mixture of extremely high-frequency
    and dead filters, and that its stride-2 second convolution was
    introducing aliasing artefacts.  Both layers were tightened
    accordingly — Conv1 became :math:`7\times7` stride-2, Conv2 kept
    :math:`5\times5` but with stride 2 — while the remaining layers
    retain AlexNet's topology.

    Parameters
    ----------
    config : ZFNetConfig
        Frozen architecture spec.  Use :func:`zfnet` for the paper-cited
        configuration; pass a custom config to override input channel
        count or dropout strength.

    Attributes
    ----------
    config : ZFNetConfig
        Stored copy of the config that built this model.
    features : nn.Sequential
        The five conv blocks with :class:`~lucid.nn.LocalResponseNorm`
        after blocks 1 and 2 and :math:`3\times3` max-pools after blocks
        1, 2, 5 — see :func:`_build_features` for the exact layer chain.
    avgpool : nn.AdaptiveAvgPool2d
        Global pool down to a :math:`6\times6` spatial map so the
        backbone produces a fixed-size feature regardless of input
        resolution.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin`.

    Notes
    -----
    From Zeiler & Fergus, "Visualizing and Understanding Convolutional
    Networks", ECCV 2014.  The paper reports a single-model top-5
    ImageNet validation error of 11.7%, an improvement of roughly four
    percentage points over the published AlexNet number, achieved with
    essentially the same parameter and FLOP budget.  More important
    than the accuracy gain was the *methodology*: ZFNet popularised
    using interpretability tooling to diagnose and fix early-layer
    problems, an idea that propagates directly to every later vision-
    saliency / Grad-CAM-style analysis.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.zfnet import zfnet
    >>> backbone = zfnet()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 256, 6, 6)
    (2, 256, 6, 6)
    """

    config_class: ClassVar[type[ZFNetConfig]] = ZFNetConfig
    base_model_prefix: ClassVar[str] = "zfnet"

    def __init__(self, config: ZFNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=96, reduction=4),
            FeatureInfo(stage=2, num_channels=256, reduction=8),
            FeatureInfo(stage=3, num_channels=384, reduction=16),
            FeatureInfo(stage=4, num_channels=384, reduction=16),
            FeatureInfo(stage=5, num_channels=256, reduction=32),
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
# ZFNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ZFNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    r"""ZFNet with two 4096-dim fully-connected layers and a linear classifier.

    Combines a :class:`ZFNet` convolutional backbone with the
    AlexNet-style classifier head: FC6 (256·6·6 → 4096), FC7
    (4096 → 4096), and a final linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    :class:`~lucid.nn.Dropout` with ``config.dropout`` (0.5 in the
    paper) is applied after both ReLU activations.  When ``labels`` are
    supplied to :meth:`forward`, cross-entropy loss is returned
    alongside the logits.

    Parameters
    ----------
    config : ZFNetConfig
        Architecture spec.  Use :func:`zfnet_cls` for the paper-cited
        ImageNet-1k configuration.

    Attributes
    ----------
    config : ZFNetConfig
        Stored copy of the config that built this model.
    features, avgpool
        Same backbone components as on :class:`ZFNet`.
    fc6, fc7 : nn.Linear
        The two 4096-dim hidden fully-connected layers.
    drop6, drop7 : nn.Dropout
        Dropout layers applied after each ReLU in the hidden stack,
        controlled by ``config.dropout``.
    classifier : nn.Module
        Final linear projection 4096 → ``num_classes``.

    Notes
    -----
    From Zeiler & Fergus, "Visualizing and Understanding Convolutional
    Networks", ECCV 2014, §3.  Top-5 ImageNet-1k validation error in
    the paper is 11.7%.  The classifier head is identical to AlexNet's
    by design — ZFNet's contribution lives entirely in the first two
    convolutional layers.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.zfnet import zfnet_cls
    >>> model = zfnet_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)
    """

    config_class: ClassVar[type[ZFNetConfig]] = ZFNetConfig
    base_model_prefix: ClassVar[str] = "zfnet"

    def __init__(self, config: ZFNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.drop6 = nn.Dropout(p=config.dropout)
        self.drop7 = nn.Dropout(p=config.dropout)
        self._build_classifier(4096, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.features(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        x = cast(Tensor, self.drop6(F.relu(cast(Tensor, self.fc6(x)))))
        x = cast(Tensor, self.drop7(F.relu(cast(Tensor, self.fc7(x)))))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
