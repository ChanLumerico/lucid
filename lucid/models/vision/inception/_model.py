"""Inception v3 backbone and classifier (Szegedy et al., 2015).

Paper: "Rethinking the Inception Architecture for Computer Vision"

Architecture overview (299×299 input):
    Stem   : Conv3×3-s2 → Conv3×3 → Conv3×3-p1 → MaxPool-s2
             → Conv1×1 → Conv3×3 → MaxPool-s2
    InceptionA × 3: 4-branch concat (pool_features=32, 32, 64)
    InceptionB (Reduction-A): 3-branch, stride-2 reduction
    InceptionC × 4: factorized n×1 / 1×n blocks (n=7)
    InceptionD (Reduction-B): 3-branch, stride-2 reduction
    InceptionE × 2: expanded branch with 1×3 / 3×1 splits
    Auxiliary classifier attaches after the last 17×17 stage (InceptionC[3] = Mixed_6e)
    Head: AdaptiveAvgPool(1×1) → Dropout(0.5) → FC(2048, num_classes)
"""

from dataclasses import dataclass
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput
from lucid.models.vision.inception._config import InceptionConfig

# ---------------------------------------------------------------------------
# Shared Conv-BN-ReLU helper
# ---------------------------------------------------------------------------


class _ConvBnReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.relu(cast(Tensor, self.bn(cast(Tensor, self.conv(x)))))


# ---------------------------------------------------------------------------
# Inception A — 3× repeated with varying pool_features
# ---------------------------------------------------------------------------


class _InceptionA(nn.Module):
    """Inception-A block (factorized 5×5 into two 3×3)."""

    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        # branch1: 1×1
        self.branch1 = _ConvBnReLU(in_channels, 64, 1)
        # branch2: 1×1 → 5×5 (implemented as two 3×3)
        self.branch2_a = _ConvBnReLU(in_channels, 48, 1)
        self.branch2_b = _ConvBnReLU(48, 64, 5, padding=2)
        # branch3: 1×1 → 3×3 → 3×3
        self.branch3_a = _ConvBnReLU(in_channels, 64, 1)
        self.branch3_b = _ConvBnReLU(64, 96, 3, padding=1)
        self.branch3_c = _ConvBnReLU(96, 96, 3, padding=1)
        # branch4: AvgPool → 1×1
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, pool_features, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
        b3 = cast(
            Tensor,
            self.branch3_c(
                cast(Tensor, self.branch3_b(cast(Tensor, self.branch3_a(x))))
            ),
        )
        b4 = cast(Tensor, self.branch4_conv(cast(Tensor, self.branch4_pool(x))))
        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Inception B — Reduction-A (stride-2 reduction 35×35 → 17×17)
# ---------------------------------------------------------------------------


class _InceptionB(nn.Module):
    """Reduction-A block: reduces 35×35 → 17×17."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 3×3 stride=2
        self.branch1 = _ConvBnReLU(in_channels, 384, 3, stride=2)
        # branch2: 1×1 → 3×3 → 3×3 stride=2
        self.branch2_a = _ConvBnReLU(in_channels, 64, 1)
        self.branch2_b = _ConvBnReLU(64, 96, 3, padding=1)
        self.branch2_c = _ConvBnReLU(96, 96, 3, stride=2)
        # branch3: MaxPool stride=2 (passthrough)
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(
            Tensor,
            self.branch2_c(
                cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
            ),
        )
        b3 = cast(Tensor, self.branch3(x))
        return lucid.cat([b1, b2, b3], dim=1)


# ---------------------------------------------------------------------------
# Inception C — 4× factorized n×1 and 1×n (n=7)
# ---------------------------------------------------------------------------


class _InceptionC(nn.Module):
    """Inception-C block with 1×n / n×1 factorization (n=7)."""

    def __init__(self, in_channels: int, channels_7x7: int) -> None:
        super().__init__()
        c7 = channels_7x7
        # branch1: 1×1
        self.branch1 = _ConvBnReLU(in_channels, 192, 1)
        # branch2: 1×1 → 1×7 → 7×1
        self.branch2_a = _ConvBnReLU(in_channels, c7, 1)
        self.branch2_b = nn.Conv2d(c7, c7, (1, 7), padding=(0, 3), bias=False)
        self.branch2_b_bn = nn.BatchNorm2d(c7)
        self.branch2_c = nn.Conv2d(c7, 192, (7, 1), padding=(3, 0), bias=False)
        self.branch2_c_bn = nn.BatchNorm2d(192)
        # branch3: 1×1 → 7×1 → 1×7 → 7×1 → 1×7
        self.branch3_a = _ConvBnReLU(in_channels, c7, 1)
        self.branch3_b = nn.Conv2d(c7, c7, (7, 1), padding=(3, 0), bias=False)
        self.branch3_b_bn = nn.BatchNorm2d(c7)
        self.branch3_c = nn.Conv2d(c7, c7, (1, 7), padding=(0, 3), bias=False)
        self.branch3_c_bn = nn.BatchNorm2d(c7)
        self.branch3_d = nn.Conv2d(c7, c7, (7, 1), padding=(3, 0), bias=False)
        self.branch3_d_bn = nn.BatchNorm2d(c7)
        self.branch3_e = nn.Conv2d(c7, 192, (1, 7), padding=(0, 3), bias=False)
        self.branch3_e_bn = nn.BatchNorm2d(192)
        # branch4: AvgPool → 1×1
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, 192, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))

        t = cast(Tensor, self.branch2_a(x))
        t = F.relu(cast(Tensor, self.branch2_b_bn(cast(Tensor, self.branch2_b(t)))))
        b2 = F.relu(cast(Tensor, self.branch2_c_bn(cast(Tensor, self.branch2_c(t)))))

        t = cast(Tensor, self.branch3_a(x))
        t = F.relu(cast(Tensor, self.branch3_b_bn(cast(Tensor, self.branch3_b(t)))))
        t = F.relu(cast(Tensor, self.branch3_c_bn(cast(Tensor, self.branch3_c(t)))))
        t = F.relu(cast(Tensor, self.branch3_d_bn(cast(Tensor, self.branch3_d(t)))))
        b3 = F.relu(cast(Tensor, self.branch3_e_bn(cast(Tensor, self.branch3_e(t)))))

        b4 = cast(Tensor, self.branch4_conv(cast(Tensor, self.branch4_pool(x))))

        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Inception D — Reduction-B (stride-2 reduction 17×17 → 8×8)
# ---------------------------------------------------------------------------


class _InceptionD(nn.Module):
    """Reduction-B block: reduces 17×17 → 8×8."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 1×1 → 3×3 stride=2
        self.branch1_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch1_b = _ConvBnReLU(192, 320, 3, stride=2)
        # branch2: 1×1 → 1×7 → 7×1 → 3×3 stride=2
        self.branch2_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch2_b = nn.Conv2d(192, 192, (1, 7), padding=(0, 3), bias=False)
        self.branch2_b_bn = nn.BatchNorm2d(192)
        self.branch2_c = nn.Conv2d(192, 192, (7, 1), padding=(3, 0), bias=False)
        self.branch2_c_bn = nn.BatchNorm2d(192)
        self.branch2_d = _ConvBnReLU(192, 192, 3, stride=2)
        # branch3: MaxPool stride=2
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1_b(cast(Tensor, self.branch1_a(x))))

        t = cast(Tensor, self.branch2_a(x))
        t = F.relu(cast(Tensor, self.branch2_b_bn(cast(Tensor, self.branch2_b(t)))))
        t = F.relu(cast(Tensor, self.branch2_c_bn(cast(Tensor, self.branch2_c(t)))))
        b2 = cast(Tensor, self.branch2_d(t))

        b3 = cast(Tensor, self.branch3(x))
        return lucid.cat([b1, b2, b3], dim=1)


# ---------------------------------------------------------------------------
# Inception E — 2× expanded branches with 1×3/3×1 splits
# ---------------------------------------------------------------------------


class _InceptionE(nn.Module):
    """Inception-E block: expanded branches with parallel 1×3 / 3×1 convs."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 1×1
        self.branch1 = _ConvBnReLU(in_channels, 320, 1)
        # branch2: 1×1(384) → [1×3(384), 3×1(384)] concat
        self.branch2_a = _ConvBnReLU(in_channels, 384, 1)
        self.branch2_b1 = nn.Conv2d(384, 384, (1, 3), padding=(0, 1), bias=False)
        self.branch2_b1_bn = nn.BatchNorm2d(384)
        self.branch2_b2 = nn.Conv2d(384, 384, (3, 1), padding=(1, 0), bias=False)
        self.branch2_b2_bn = nn.BatchNorm2d(384)
        # branch3: 1×1(448) → 3×3(384) → [1×3(384), 3×1(384)] concat
        self.branch3_a = _ConvBnReLU(in_channels, 448, 1)
        self.branch3_b = _ConvBnReLU(448, 384, 3, padding=1)
        self.branch3_c1 = nn.Conv2d(384, 384, (1, 3), padding=(0, 1), bias=False)
        self.branch3_c1_bn = nn.BatchNorm2d(384)
        self.branch3_c2 = nn.Conv2d(384, 384, (3, 1), padding=(1, 0), bias=False)
        self.branch3_c2_bn = nn.BatchNorm2d(384)
        # branch4: AvgPool → 1×1(192)
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, 192, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))

        t2 = cast(Tensor, self.branch2_a(x))
        b2a = F.relu(
            cast(Tensor, self.branch2_b1_bn(cast(Tensor, self.branch2_b1(t2))))
        )
        b2b = F.relu(
            cast(Tensor, self.branch2_b2_bn(cast(Tensor, self.branch2_b2(t2))))
        )
        b2 = lucid.cat([b2a, b2b], dim=1)

        t3 = cast(Tensor, self.branch3_b(cast(Tensor, self.branch3_a(x))))
        b3a = F.relu(
            cast(Tensor, self.branch3_c1_bn(cast(Tensor, self.branch3_c1(t3))))
        )
        b3b = F.relu(
            cast(Tensor, self.branch3_c2_bn(cast(Tensor, self.branch3_c2(t3))))
        )
        b3 = lucid.cat([b3a, b3b], dim=1)

        b4 = cast(Tensor, self.branch4_conv(cast(Tensor, self.branch4_pool(x))))

        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Auxiliary classifier
# ---------------------------------------------------------------------------


class _InceptionAux(nn.Module):
    """Auxiliary classifier for Inception v3 (attaches after InceptionC[3]).

    Matches the reference-framework / timm structure:
      AvgPool(5,s=3) → Conv(in→128, 1×1) → Conv(128→768, 5×5) → FC(768, num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.avgpool = nn.AvgPool2d(5, stride=3)
        self.conv0 = _ConvBnReLU(in_channels, 128, 1)
        self.conv1 = _ConvBnReLU(128, 768, 5)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.avgpool(x))
        x = cast(Tensor, self.conv0(x))
        x = cast(Tensor, self.conv1(x))
        x = cast(Tensor, self.adapt_pool(x))
        x = x.flatten(1)
        return cast(Tensor, self.fc(x))


# ---------------------------------------------------------------------------
# Inception v3 output dataclass
# ---------------------------------------------------------------------------


@dataclass
class InceptionV3Output:
    r"""Structured forward output for :class:`InceptionV3ForImageClassification`.

    Carries the main classifier logits and the optional auxiliary
    classifier logits emitted during training only.  The auxiliary head
    in Inception v3 attaches after the last 17x17 stage
    (``inception_c3`` / Mixed_6e in the paper diagram) and is only
    active when both ``config.aux_logits=True`` and the model is in
    training mode.

    Parameters
    ----------
    logits : Tensor
        Main classifier output of shape ``(B, num_classes)``.
    aux_logits : Tensor or None, optional, default=None
        Logits from the auxiliary classifier of shape
        ``(B, num_classes)``; ``None`` at inference or when
        ``aux_logits=False``.
    loss : Tensor or None, optional, default=None
        Cross-entropy loss with auxiliary term (weight 0.4) when labels
        were passed to :meth:`forward`; ``None`` otherwise.

    Notes
    -----
    From Szegedy et al., "Rethinking the Inception Architecture for
    Computer Vision", CVPR 2016, §6.  Training loss with the auxiliary
    head is

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{main}}
            + 0.4 \cdot \mathcal{L}_{\text{aux}}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception import inception_v3_cls
    >>> model = inception_v3_cls(aux_logits=True).eval()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> out = model(x)
    >>> out.aux_logits is None   # inactive at eval
    True
    """

    logits: Tensor
    aux_logits: Tensor | None = None
    loss: Tensor | None = None


# ---------------------------------------------------------------------------
# Stem builder
# ---------------------------------------------------------------------------


def _build_inception_stem(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        _ConvBnReLU(in_channels, 32, 3, stride=2),
        _ConvBnReLU(32, 32, 3),
        _ConvBnReLU(32, 64, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        _ConvBnReLU(64, 80, 1),
        _ConvBnReLU(80, 192, 3),
        nn.MaxPool2d(3, stride=2),
    )


# ---------------------------------------------------------------------------
# InceptionV3 backbone (task="base")
# ---------------------------------------------------------------------------


class InceptionV3(PretrainedModel, BackboneMixin):
    r"""Inception v3 feature-extracting backbone.

    Implements the topology from Szegedy et al., "Rethinking the
    Inception Architecture for Computer Vision", CVPR 2016: a deep
    pre-Inception stem (3×Conv → MaxPool → 2×Conv → MaxPool), three
    :class:`_InceptionA` blocks at :math:`35\times35`, a Reduction-A
    block (:class:`_InceptionB`) down to :math:`17\times17`, four
    :class:`_InceptionC` blocks with factorised :math:`1\times7` and
    :math:`7\times1` convolutions, a Reduction-B block
    (:class:`_InceptionD`) down to :math:`8\times8`, two
    :class:`_InceptionE` blocks with the expanded
    :math:`1\times3`/:math:`3\times1` parallel branches, and a final
    :class:`~lucid.nn.AdaptiveAvgPool2d` to :math:`1\times1`.  Designed
    for :math:`299\times299` RGB inputs.

    Parameters
    ----------
    config : InceptionConfig
        Frozen architecture spec.  Use :func:`inception_v3` for the
        paper-cited configuration.  ``aux_logits`` only affects the
        classifier variant.

    Attributes
    ----------
    config : InceptionConfig
        Stored copy of the config that built this model.
    stem : nn.Sequential
        The pre-Inception conv stack.
    inception_a0, inception_a1, inception_a2 : _InceptionA
        Three Inception-A blocks at :math:`35\times35` (factorised
        :math:`5\times5` into two :math:`3\times3`).
    reduction_a : _InceptionB
        Reduction-A block reducing spatial size :math:`35\times35
        \to 17\times17`.
    inception_c0, inception_c1, inception_c2, inception_c3 : _InceptionC
        Four Inception-C blocks at :math:`17\times17` with
        :math:`1\times7` / :math:`7\times1` factorisation.
    reduction_b : _InceptionD
        Reduction-B block reducing spatial size :math:`17\times17
        \to 8\times8`.
    inception_e0, inception_e1 : _InceptionE
        Two Inception-E blocks at :math:`8\times8` with expanded
        :math:`1\times3` / :math:`3\times1` parallel branches.
    avgpool : nn.AdaptiveAvgPool2d
        Global average pool to :math:`1\times1`.
    feature_info : list[FeatureInfo]
        Per-stage descriptor (channels + reduction factor) exposed via
        :class:`BackboneMixin`.

    Notes
    -----
    The headline factorisation idea is to replace an
    :math:`n\times n` convolution with an :math:`n\times1` followed by
    a :math:`1\times n` convolution: for :math:`n = 7` this trades
    :math:`49 C^2` parameters for :math:`14 C^2` while adding an extra
    ReLU and matching the original receptive field along a separable
    manifold.  Inception v3 uses three block topologies (A, B, C) at
    three spatial resolutions (35×35, 17×17, 8×8), each tailored to
    the receptive-field budget at that stage.  Approximately 23.8 M
    parameters (without auxiliary head); achieves a top-5 ImageNet
    error of 3.5% on the validation set.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception import inception_v3
    >>> backbone = inception_v3()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> out = backbone(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 1, 1)
    (1, 2048, 1, 1)
    """

    config_class: ClassVar[type[InceptionConfig]] = InceptionConfig
    base_model_prefix: ClassVar[str] = "inception_v3"

    def __init__(self, config: InceptionConfig) -> None:
        super().__init__(config)
        self.stem = _build_inception_stem(config.in_channels)

        # InceptionA × 3 (pool_features = 32, 64, 64)
        # InceptionA always outputs 64+64+96+pool_features channels
        # a0: 192→256 (pool=32), a1: 256→288 (pool=64), a2: 288→288 (pool=64)
        self.inception_a0 = _InceptionA(192, pool_features=32)
        self.inception_a1 = _InceptionA(256, pool_features=64)
        self.inception_a2 = _InceptionA(288, pool_features=64)

        # Reduction-A (InceptionB) — input: 288 channels = 64+64+96+64
        self.reduction_a = _InceptionB(288)

        # InceptionC × 4 (channels_7x7 = 128, 160, 160, 192)
        self.inception_c0 = _InceptionC(768, channels_7x7=128)
        self.inception_c1 = _InceptionC(768, channels_7x7=160)
        self.inception_c2 = _InceptionC(768, channels_7x7=160)
        self.inception_c3 = _InceptionC(768, channels_7x7=192)

        # Reduction-B (InceptionD)
        self.reduction_b = _InceptionD(768)

        # InceptionE × 2
        self.inception_e0 = _InceptionE(1280)
        self.inception_e1 = _InceptionE(2048)

        # Final pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=288, reduction=8),
            FeatureInfo(stage=2, num_channels=768, reduction=16),
            FeatureInfo(stage=3, num_channels=2048, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.inception_a0(x))
        x = cast(Tensor, self.inception_a1(x))
        x = cast(Tensor, self.inception_a2(x))
        x = cast(Tensor, self.reduction_a(x))
        x = cast(Tensor, self.inception_c0(x))
        x = cast(Tensor, self.inception_c1(x))
        x = cast(Tensor, self.inception_c2(x))
        x = cast(Tensor, self.inception_c3(x))
        x = cast(Tensor, self.reduction_b(x))
        x = cast(Tensor, self.inception_e0(x))
        x = cast(Tensor, self.inception_e1(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# InceptionV3 for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class InceptionV3ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    r"""Inception v3 image classifier with optional auxiliary classifier.

    Combines an :class:`InceptionV3` backbone with a global-average-pool
    + :class:`~lucid.nn.Dropout` (``p=config.dropout``, default 0.5) +
    :class:`~lucid.nn.Linear` head producing ``config.num_classes``
    logits.  When ``config.aux_logits=True``, an auxiliary classifier
    (:class:`_InceptionAux`) attaches after the last
    :math:`17\times17` Inception-C block and contributes a 0.4-weighted
    cross-entropy term during training.

    Parameters
    ----------
    config : InceptionConfig
        Architecture spec.  Use :func:`inception_v3_cls` for the
        timm-compatible default (no auxiliary head).  Pass
        ``aux_logits=True`` to enable the auxiliary classifier — used
        during training to combat vanishing gradients and provide
        intermediate supervision.

    Attributes
    ----------
    config : InceptionConfig
        Stored copy of the config that built this model.
    stem, inception_a0, ..., inception_e1, reduction_a, reduction_b
        Same backbone components as on :class:`InceptionV3`.
    avgpool : nn.AdaptiveAvgPool2d
        Final global average pool to :math:`1\times1`.
    drop : nn.Dropout
        Dropout applied before the main classifier
        (``p=config.dropout``, 0.5 in the paper).
    classifier : nn.Module
        Final linear projection 2048 → ``num_classes``.
    aux : nn.Module
        Auxiliary classifier (:class:`_InceptionAux`) when
        ``config.aux_logits=True``; otherwise
        :class:`~lucid.nn.Identity`.

    Notes
    -----
    From Szegedy et al., "Rethinking the Inception Architecture for
    Computer Vision", CVPR 2016, §6.  Total loss with auxiliary head is

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{main}}
            + 0.4 \cdot \mathcal{L}_{\text{aux}}.

    The paper also introduces *label smoothing* — replacing the
    one-hot target with :math:`(1 - \epsilon)\,\mathbf{1}_y +
    \epsilon/K` — although that is a training-loop concern handled
    outside this module.  Final top-5 ImageNet validation error in the
    paper is 3.5% with ≈24 M parameters.

    Examples
    --------
    Inference path (auxiliary head ignored even if enabled):

    >>> import lucid
    >>> from lucid.models.vision.inception import inception_v3_cls
    >>> model = inception_v3_cls().eval()
    >>> x = lucid.randn(2, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """

    config_class: ClassVar[type[InceptionConfig]] = InceptionConfig
    base_model_prefix: ClassVar[str] = "inception_v3"

    def __init__(self, config: InceptionConfig) -> None:
        super().__init__(config)
        self.stem = _build_inception_stem(config.in_channels)

        # InceptionA × 3
        # a0: 192→256 (pool=32), a1: 256→288 (pool=64), a2: 288→288 (pool=64)
        self.inception_a0 = _InceptionA(192, pool_features=32)
        self.inception_a1 = _InceptionA(256, pool_features=64)
        self.inception_a2 = _InceptionA(288, pool_features=64)

        # Reduction-A (input: 288 channels = 64+64+96+64)
        self.reduction_a = _InceptionB(288)

        # InceptionC × 4
        self.inception_c0 = _InceptionC(768, channels_7x7=128)
        self.inception_c1 = _InceptionC(768, channels_7x7=160)
        self.inception_c2 = _InceptionC(768, channels_7x7=160)
        self.inception_c3 = _InceptionC(768, channels_7x7=192)

        # Reduction-B
        self.reduction_b = _InceptionD(768)

        # InceptionE × 2
        self.inception_e0 = _InceptionE(1280)
        self.inception_e1 = _InceptionE(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=config.dropout)
        self._build_classifier(2048, config.num_classes)

        # Auxiliary classifier — paper §6 / Fig.10 attaches after the last
        # 17×17 stage (Mixed_6e = inception_c3), not after c1.
        if config.aux_logits:
            self.aux: nn.Module = _InceptionAux(768, config.num_classes)
        else:
            self.aux = nn.Identity()

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> InceptionV3Output:
        cfg = self.config
        assert isinstance(cfg, InceptionConfig)
        use_aux = cfg.aux_logits and self.training

        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.inception_a0(x))
        x = cast(Tensor, self.inception_a1(x))
        x = cast(Tensor, self.inception_a2(x))
        x = cast(Tensor, self.reduction_a(x))
        x = cast(Tensor, self.inception_c0(x))
        x = cast(Tensor, self.inception_c1(x))
        x = cast(Tensor, self.inception_c2(x))
        x = cast(Tensor, self.inception_c3(x))

        aux_out: Tensor | None = None
        if use_aux and isinstance(self.aux, _InceptionAux):
            aux_out = cast(Tensor, self.aux(x))

        x = cast(Tensor, self.reduction_b(x))
        x = cast(Tensor, self.inception_e0(x))
        x = cast(Tensor, self.inception_e1(x))
        x = cast(Tensor, self.avgpool(x))
        x = cast(Tensor, self.drop(x.flatten(1)))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux_out is not None:
                loss = loss + 0.4 * F.cross_entropy(aux_out, labels)

        return InceptionV3Output(logits=logits, aux_logits=aux_out, loss=loss)
