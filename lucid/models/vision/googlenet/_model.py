"""GoogLeNet (Inception v1) backbone and classifier (Szegedy et al., 2014).

Paper: "Going Deeper with Convolutions"

Architecture overview:
    Stem   : Conv7×7-s2 → MaxPool3×3-s2 → Conv1×1 → Conv3×3 → MaxPool3×3-s2
    Stage 3: Inception(3a) → Inception(3b) → MaxPool3×3-s2
    Stage 4: Inception(4a)[→aux1] → (4b) → (4c) → (4d)[→aux2] → (4e)
             → MaxPool3×3-s2
    Stage 5: Inception(5a) → Inception(5b)
    Head   : AdaptiveAvgPool(1×1) → Dropout(0.4) → FC(1024, num_classes)

Each Inception module is a four-branch parallel block:
    branch1 : 1×1 conv
    branch2 : 1×1 conv → 3×3 conv
    branch3 : 1×1 conv → 5×5 conv
    branch4 : 3×3 max pool → 1×1 conv
    → channel-wise concat

The two auxiliary classifiers (training only) attach at Inception 4a and 4d:
    AvgPool5×5-s3 → Conv1×1(128) → FC(1024) → Dropout(0.7) → FC(num_classes)
"""

from dataclasses import dataclass
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models.vision.googlenet._config import GoogLeNetConfig

# ---------------------------------------------------------------------------
# Inception module
# ---------------------------------------------------------------------------


class _InceptionModule(nn.Module):
    """Single Inception block with four parallel branches."""

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        out_3x3_reduce: int,
        out_3x3: int,
        out_5x5_reduce: int,
        out_5x5: int,
        out_pool_proj: int,
    ) -> None:
        super().__init__()
        # Branch 1: 1×1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, 1),
            nn.ReLU(inplace=True),
        )
        # Branch 2: 1×1 → 3×3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3_reduce, out_3x3, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Branch 3: 1×1 → 5×5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5_reduce, out_5x5, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        # Branch 4: 3×3 max pool → 1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool_proj, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2(x))
        b3 = cast(Tensor, self.branch3(x))
        b4 = cast(Tensor, self.branch4(x))
        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Auxiliary classifier (training only)
# ---------------------------------------------------------------------------


class _AuxClassifier(nn.Module):
    """Auxiliary classifier attached at Inception 4a and 4d."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, 1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.avgpool(x))
        x = F.relu(cast(Tensor, self.conv(x)))
        x = x.flatten(1)
        x = cast(Tensor, self.drop(F.relu(cast(Tensor, self.fc1(x)))))
        return cast(Tensor, self.fc2(x))


# ---------------------------------------------------------------------------
# GoogLeNet output dataclass
# ---------------------------------------------------------------------------


@dataclass
class GoogLeNetOutput:
    """GoogLeNet output — includes optional auxiliary logits for training."""

    logits: Tensor
    aux_logits1: Tensor | None = None
    aux_logits2: Tensor | None = None
    loss: Tensor | None = None


# ---------------------------------------------------------------------------
# Shared stem builder
# ---------------------------------------------------------------------------


def _build_stem(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        nn.Conv2d(64, 64, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 192, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
    )


# Inception module specs: (out_1x1, out_3x3r, out_3x3, out_5x5r, out_5x5, out_pool)
_INCEPTION_SPECS: list[tuple[int, int, int, int, int, int, int]] = [
    # in_ch, 1x1, 3x3r, 3x3, 5x5r, 5x5, pool
    (192, 64, 96, 128, 16, 32, 32),  # 3a → 256
    (256, 128, 128, 192, 32, 96, 64),  # 3b → 480
    # after maxpool(3×3 s2)
    (480, 192, 96, 208, 16, 48, 64),  # 4a → 512  [aux1 here]
    (512, 160, 112, 224, 24, 64, 64),  # 4b → 512
    (512, 128, 128, 256, 24, 64, 64),  # 4c → 512
    (512, 112, 144, 288, 32, 64, 64),  # 4d → 528  [aux2 here]
    (528, 256, 160, 320, 32, 128, 128),  # 4e → 832
    # after maxpool(3×3 s2)
    (832, 256, 160, 320, 32, 128, 128),  # 5a → 832
    (832, 384, 192, 384, 48, 128, 128),  # 5b → 1024
]


def _make_inception(spec: tuple[int, int, int, int, int, int, int]) -> _InceptionModule:
    in_ch, o1, o3r, o3, o5r, o5, op = spec
    return _InceptionModule(in_ch, o1, o3r, o3, o5r, o5, op)


# ---------------------------------------------------------------------------
# GoogLeNet backbone  (task="base")
# ---------------------------------------------------------------------------


class GoogLeNet(PretrainedModel, BackboneMixin):
    """GoogLeNet feature extractor — outputs 1024-dim spatial map from Inception 5b.

    ``forward_features`` returns shape ``(B, 1024, 1, 1)`` for 224×224 inputs
    after AdaptiveAvgPool2d(1×1).
    Auxiliary classifiers are not included in the backbone.
    """

    config_class: ClassVar[type[GoogLeNetConfig]] = GoogLeNetConfig
    base_model_prefix: ClassVar[str] = "googlenet"

    def __init__(self, config: GoogLeNetConfig) -> None:
        super().__init__(config)
        self.stem = _build_stem(config.in_channels)

        specs = _INCEPTION_SPECS
        self.inception3a = _make_inception(specs[0])
        self.inception3b = _make_inception(specs[1])
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = _make_inception(specs[2])
        self.inception4b = _make_inception(specs[3])
        self.inception4c = _make_inception(specs[4])
        self.inception4d = _make_inception(specs[5])
        self.inception4e = _make_inception(specs[6])
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = _make_inception(specs[7])
        self.inception5b = _make_inception(specs[8])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=3, num_channels=480, reduction=8),
            FeatureInfo(stage=4, num_channels=832, reduction=16),
            FeatureInfo(stage=5, num_channels=1024, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.inception3a(x))
        x = cast(Tensor, self.inception3b(x))
        x = cast(Tensor, self.maxpool3(x))
        x = cast(Tensor, self.inception4a(x))
        x = cast(Tensor, self.inception4b(x))
        x = cast(Tensor, self.inception4c(x))
        x = cast(Tensor, self.inception4d(x))
        x = cast(Tensor, self.inception4e(x))
        x = cast(Tensor, self.maxpool4(x))
        x = cast(Tensor, self.inception5a(x))
        x = cast(Tensor, self.inception5b(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> GoogLeNetOutput:  # type: ignore[override]
        return GoogLeNetOutput(logits=self.forward_features(x))


# ---------------------------------------------------------------------------
# GoogLeNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class GoogLeNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """GoogLeNet with Dropout + FC head and (optionally) auxiliary classifiers.

    During training (``model.train()``) and when ``config.aux_logits=True``,
    ``forward`` returns ``GoogLeNetOutput`` with ``aux_logits1`` /
    ``aux_logits2`` populated.  At inference these are ``None``.
    """

    config_class: ClassVar[type[GoogLeNetConfig]] = GoogLeNetConfig
    base_model_prefix: ClassVar[str] = "googlenet"

    def __init__(self, config: GoogLeNetConfig) -> None:
        super().__init__(config)
        self.stem = _build_stem(config.in_channels)

        specs = _INCEPTION_SPECS
        self.inception3a = _make_inception(specs[0])
        self.inception3b = _make_inception(specs[1])
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = _make_inception(specs[2])
        self.inception4b = _make_inception(specs[3])
        self.inception4c = _make_inception(specs[4])
        self.inception4d = _make_inception(specs[5])
        self.inception4e = _make_inception(specs[6])
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = _make_inception(specs[7])
        self.inception5b = _make_inception(specs[8])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=config.dropout)
        self._build_classifier(1024, config.num_classes)

        # Auxiliary classifiers (attached at 4a → 512ch, 4d → 528ch)
        if config.aux_logits:
            self.aux1: nn.Module = _AuxClassifier(
                512, config.num_classes, config.aux_dropout
            )
            self.aux2: nn.Module = _AuxClassifier(
                528, config.num_classes, config.aux_dropout
            )
        else:
            self.aux1 = nn.Identity()
            self.aux2 = nn.Identity()

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> GoogLeNetOutput:
        cfg = self.config
        assert isinstance(cfg, GoogLeNetConfig)
        use_aux = cfg.aux_logits and self.training

        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.inception3a(x))
        x = cast(Tensor, self.inception3b(x))
        x = cast(Tensor, self.maxpool3(x))

        x = cast(Tensor, self.inception4a(x))
        aux1: Tensor | None = None
        if use_aux and isinstance(self.aux1, _AuxClassifier):
            aux1 = cast(Tensor, self.aux1(x))

        x = cast(Tensor, self.inception4b(x))
        x = cast(Tensor, self.inception4c(x))
        x = cast(Tensor, self.inception4d(x))
        aux2: Tensor | None = None
        if use_aux and isinstance(self.aux2, _AuxClassifier):
            aux2 = cast(Tensor, self.aux2(x))

        x = cast(Tensor, self.inception4e(x))
        x = cast(Tensor, self.maxpool4(x))
        x = cast(Tensor, self.inception5a(x))
        x = cast(Tensor, self.inception5b(x))

        x = cast(Tensor, self.avgpool(x))
        x = cast(Tensor, self.drop(x.flatten(1)))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux1 is not None:
                loss = loss + 0.3 * F.cross_entropy(aux1, labels)
            if aux2 is not None:
                loss = loss + 0.3 * F.cross_entropy(aux2, labels)

        return GoogLeNetOutput(
            logits=logits, aux_logits1=aux1, aux_logits2=aux2, loss=loss
        )
