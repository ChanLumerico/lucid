"""AlexNet backbone and classifier (Krizhevsky, Sutskever & Hinton, 2012).

Paper: "ImageNet Classification with Deep Convolutional Neural Networks"
Architecture:
    Conv1 : 3→96,  11×11, stride=4, pad=2  → ReLU → LRN → MaxPool 3×3 s2
    Conv2 : 96→256,  5×5,  pad=2            → ReLU → LRN → MaxPool 3×3 s2
    Conv3 : 256→384, 3×3,  pad=1            → ReLU
    Conv4 : 384→384, 3×3,  pad=1            → ReLU
    Conv5 : 384→256, 3×3,  pad=1            → ReLU → MaxPool 3×3 s2
    AdaptiveAvgPool → 6×6
    FC6   : 256*6*6 → 4096                  → ReLU → Dropout
    FC7   : 4096    → 4096                  → ReLU → Dropout
    FC8   : 4096    → num_classes

The original paper split conv filters across two GPUs; the merged single-
stream version (as in standard implementations) is used here.

LRN (Local Response Normalisation) is kept for historical accuracy.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.alexnet._config import AlexNetConfig


def _build_features(cfg: AlexNetConfig) -> nn.Sequential:
    return nn.Sequential(
        # Block 1
        nn.Conv2d(cfg.in_channels, 96, 11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
        nn.MaxPool2d(3, stride=2),
        # Block 2
        nn.Conv2d(96, 256, 5, padding=2),
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
# AlexNet backbone  (task="base")
# ---------------------------------------------------------------------------


class AlexNet(PretrainedModel, BackboneMixin):
    r"""AlexNet feature-extracting backbone (no fully-connected head).

    Implements the five-stage convolutional trunk from Krizhevsky,
    Sutskever & Hinton, "ImageNet Classification with Deep
    Convolutional Neural Networks", NIPS 2012: an :math:`11\times11`
    stride-4 first convolution, an :math:`5\times5` second
    convolution, three :math:`3\times3` convolutions, with
    :class:`~lucid.nn.LocalResponseNorm` after the first two blocks
    and overlapping :math:`3\times3` max-pools that reduce the spatial
    size by an additional factor of 2 after blocks 1, 2, and 5.  A
    final :class:`~lucid.nn.AdaptiveAvgPool2d` collapses the feature
    map to :math:`6\times6` regardless of input resolution.  The
    original two-GPU model-parallel split is collapsed into a single
    merged stream — standard practice in modern reimplementations.

    Parameters
    ----------
    config : AlexNetConfig
        Frozen architecture spec.  Use :func:`alexnet` for the
        paper-cited single-stream configuration; pass a custom config
        to switch input channel count or to retarget the classifier
        variant.

    Attributes
    ----------
    config : AlexNetConfig
        Stored copy of the config that built this model.
    features : nn.Sequential
        The five conv blocks (Conv → ReLU → optional LRN → optional
        MaxPool) — see :func:`_build_features` for the exact ordering.
    avgpool : nn.AdaptiveAvgPool2d
        Global pool down to a :math:`6\times6` spatial map so the
        backbone produces a fixed-size feature regardless of input
        resolution.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin` for downstream decoder modules.

    Notes
    -----
    From Krizhevsky et al., NIPS 2012, §3 and Figure 2.  AlexNet's
    contribution to deep-learning history is threefold: the
    *rectified linear unit* :math:`\phi(x) = \max(0, x)` replaced
    saturating nonlinearities and cut training time by several factors;
    *dropout* with :math:`p = 0.5` regularised the 4096-dim
    fully-connected layers against overfitting on a 1.2 M-image dataset;
    and *local response normalisation*

    .. math::

        b_{x,y}^i = \frac{a_{x,y}^i}{
            \left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)}
            (a_{x,y}^j)^2 \right)^{\beta}
        }

    provided implicit lateral inhibition between feature maps —
    superseded later by :class:`~lucid.nn.BatchNorm2d`.  The total
    parameter count is approximately 60 M (≈58 M of which sit in the
    two 4096-dim fully-connected layers of the classifier variant).
    With the original ImageNet-1k training recipe AlexNet reaches a
    top-5 error of 15.3%.

    Examples
    --------
    Build the backbone and run a single forward pass:

    >>> import lucid
    >>> from lucid.models.vision.alexnet import alexnet
    >>> backbone = alexnet()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 256, 6, 6)
    (2, 256, 6, 6)

    Inspect per-stage feature descriptors:

    >>> info = backbone.feature_info
    >>> [(fi.stage, fi.num_channels, fi.reduction) for fi in info[:2]]
    [(1, 96, 4), (2, 256, 8)]
    """

    config_class: ClassVar[type[AlexNetConfig]] = AlexNetConfig
    base_model_prefix: ClassVar[str] = "alexnet"

    def __init__(self, config: AlexNetConfig) -> None:
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
# AlexNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class AlexNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    r"""AlexNet with two 4096-dim fully-connected layers and a linear classifier head.

    Combines an :class:`AlexNet` convolutional backbone with the
    paper-cited three-layer classifier head: FC6 (256·6·6 → 4096),
    FC7 (4096 → 4096), and the final linear projection to
    ``config.num_classes``.  :class:`~lucid.nn.Dropout` with
    ``config.dropout`` is applied after both ReLU activations in the
    hidden layers — these two large FC layers dominate the parameter
    count and are the main overfitting risk that dropout was introduced
    to control.  When ``labels`` are supplied to :meth:`forward`, a
    cross-entropy loss is returned alongside the logits.

    Parameters
    ----------
    config : AlexNetConfig
        Architecture spec.  Use :func:`alexnet_cls` for the paper-cited
        ImageNet-1k configuration (1000-class head, ``dropout=0.5``).

    Attributes
    ----------
    config : AlexNetConfig
        Stored copy of the config that built this model.
    features, avgpool
        Same backbone components as on :class:`AlexNet`; see that class
        for shape semantics.
    fc6, fc7 : nn.Linear
        The two hidden fully-connected layers, both projecting to 4096
        dimensions.
    drop6, drop7 : nn.Dropout
        :class:`~lucid.nn.Dropout` layers applied after each ReLU in
        the hidden FC stack, controlled by ``config.dropout``.
    classifier : nn.Module
        Final linear projection to ``num_classes``, built by
        :meth:`ClassificationHeadMixin._build_classifier`.

    Notes
    -----
    From Krizhevsky et al., NIPS 2012, §3 and Figure 2.  The two
    4096-dim hidden layers alone account for roughly 54 M of the
    network's 60 M parameters — the original rationale for *dropout*,
    which randomly zeros out half of each FC activation during training
    so that no individual co-adapted neuron is critical for any single
    decision.  Loss is the standard cross-entropy

    .. math::

        \mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N}
            \log \operatorname{softmax}(\text{logits}_n)_{\,y_n}.

    Examples
    --------
    Run inference on a batch of 224x224 RGB images:

    >>> import lucid
    >>> from lucid.models.vision.alexnet import alexnet_cls
    >>> model = alexnet_cls()
    >>> x = lucid.randn(4, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (4, 1000)
    >>> out.loss is None
    True

    Compute a training loss given integer labels:

    >>> labels = lucid.tensor([0, 1, 2, 3], dtype=lucid.int64)
    >>> out = model(x, labels=labels)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[AlexNetConfig]] = AlexNetConfig
    base_model_prefix: ClassVar[str] = "alexnet"

    def __init__(self, config: AlexNetConfig) -> None:
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
