"""Xception backbone and classifier (Chollet, 2017).

Paper: "Xception: Deep Learning with Depthwise Separable Convolutions"

Architecture overview (299×299 input):
    Stem:
        conv1 + bn1 → ReLU  (Conv 3×3-s2, 32 ch)
        conv2 + bn2 → ReLU  (Conv 3×3, 64 ch)
    Entry flow (3 blocks — block1, block2, block3):
        blockN.rep  — Sequential of ReLU/SepConv/BN ops
        blockN.skip — 1×1 Conv (channel projection)
        blockN.skipbn — BN on skip
    Middle flow (8 blocks — block4…block11):
        blockN.rep  — Sequential of 3× (ReLU+SepConv+BN)
    Exit flow (block12 + conv3/bn3 + conv4/bn4):
        block12.rep  — Sequential (SepConv(728)+SepConv(1024)+MaxPool)
        block12.skip / skipbn
        conv3 + bn3  → ReLU  (SepConv 1536)
        conv4 + bn4  → ReLU  (SepConv 2048)
    Head:
        AdaptiveAvgPool(1×1) → Dropout → fc

SepConv sub-module attribute names (timm layout):
    conv1  — depthwise Conv2d (groups=in_ch)
    pointwise — pointwise Conv2d (1×1)
    (BN stored separately in the parent Sequential, not inside _SepConv)
"""

from dataclasses import dataclass
from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput
from lucid.models.vision.xception._config import XceptionConfig

# ---------------------------------------------------------------------------
# Low-level SepConv primitive matching timm attribute names
# ---------------------------------------------------------------------------


class _SepConvOp(nn.Module):
    """Depthwise + pointwise conv (no BN, no activation).

    Attribute names match timm legacy_xception layout:
      self.conv1      — depthwise Conv2d (groups=in_channels)
      self.pointwise  — pointwise Conv2d (1×1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.conv1(x))
        return cast(Tensor, self.pointwise(x))


# ---------------------------------------------------------------------------
# Entry flow block (block1 / block2 / block3)
# ---------------------------------------------------------------------------


def _make_entry_block(
    in_channels: int,
    out_channels: int,
    *,
    activate_first: bool,
) -> nn.Module:
    """Return a block matching timm entry-flow key layout.

    timm key layout for block1 (activate_first=False):
      rep.0.conv1/pointwise  — SepConv1 depthwise/pointwise
      rep.1.*                — BN after SepConv1
      rep.2                  — ReLU  (no params, index 2)
      rep.3.conv1/pointwise  — SepConv2
      rep.4.*                — BN after SepConv2
      rep.5                  — MaxPool2d
      skip                   — 1×1 Conv (channel projection)
      skipbn                 — BN on skip

    timm key layout for block2/block3 (activate_first=True):
      rep.0                  — ReLU  (no params, index 0)
      rep.1.conv1/pointwise  — SepConv1
      rep.2.*                — BN
      rep.3                  — ReLU
      rep.4.conv1/pointwise  — SepConv2
      rep.5.*                — BN
      rep.6                  — MaxPool2d
    """

    class _EntryBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ops: list[nn.Module] = []
            if activate_first:
                ops.append(nn.ReLU(inplace=True))
            # SepConv 1
            ops.append(_SepConvOp(in_channels, out_channels))
            ops.append(nn.BatchNorm2d(out_channels))
            ops.append(nn.ReLU(inplace=True))
            # SepConv 2
            ops.append(_SepConvOp(out_channels, out_channels))
            ops.append(nn.BatchNorm2d(out_channels))
            ops.append(nn.MaxPool2d(3, stride=2, padding=1))
            self.rep = nn.Sequential(*ops)
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            residual = cast(Tensor, self.skipbn(cast(Tensor, self.skip(x))))
            return cast(Tensor, self.rep(x)) + residual

    return _EntryBlock()


# ---------------------------------------------------------------------------
# Middle flow block (block4 … block11)
# ---------------------------------------------------------------------------


def _make_middle_block(channels: int) -> nn.Module:
    """Return a block matching timm middle-flow key layout.

    timm key layout (example: block4):
      block4.rep.1.conv1  block4.rep.1.pointwise  block4.rep.2.*  — SepConv1
      block4.rep.4.conv1  block4.rep.4.pointwise  block4.rep.5.*  — SepConv2
      block4.rep.7.conv1  block4.rep.7.pointwise  block4.rep.8.*  — SepConv3

    The `rep` Sequential indices:
      0 — ReLU
      1 — _SepConvOp
      2 — BN
      3 — ReLU
      4 — _SepConvOp
      5 — BN
      6 — ReLU
      7 — _SepConvOp
      8 — BN
    """

    class _MiddleBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rep = nn.Sequential(
                nn.ReLU(inplace=True),
                _SepConvOp(channels, channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                _SepConvOp(channels, channels),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                _SepConvOp(channels, channels),
                nn.BatchNorm2d(channels),
            )

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            return cast(Tensor, self.rep(x)) + x

    return _MiddleBlock()


# ---------------------------------------------------------------------------
# Exit flow block (block12)
# ---------------------------------------------------------------------------


def _make_exit_block() -> nn.Module:
    """Return a block matching timm exit-flow key layout.

    timm key layout (block12):
      block12.rep.1.conv1  block12.rep.1.pointwise  block12.rep.2.*  — SepConv(728)
      block12.rep.4.conv1  block12.rep.4.pointwise  block12.rep.5.*  — SepConv(1024)
      block12.skip         — 1×1 Conv 728→1024, stride=2
      block12.skipbn       — BN(1024)

    The `rep` Sequential indices:
      0 — ReLU
      1 — _SepConvOp(728, 728)
      2 — BN(728)
      3 — ReLU
      4 — _SepConvOp(728, 1024)
      5 — BN(1024)
      6 — MaxPool2d(3, stride=2, padding=1)
    """

    class _ExitBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rep = nn.Sequential(
                nn.ReLU(inplace=True),
                _SepConvOp(728, 728),
                nn.BatchNorm2d(728),
                nn.ReLU(inplace=True),
                _SepConvOp(728, 1024),
                nn.BatchNorm2d(1024),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
            self.skip = nn.Conv2d(728, 1024, 1, stride=2, bias=False)
            self.skipbn = nn.BatchNorm2d(1024)

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            residual = cast(Tensor, self.skipbn(cast(Tensor, self.skip(x))))
            return cast(Tensor, self.rep(x)) + residual

    return _ExitBlock()


# ---------------------------------------------------------------------------
# SepConv used in exit conv3/conv4 (wraps _SepConvOp + BN; called separately)
# ---------------------------------------------------------------------------


class _ExitSepConv(nn.Module):
    """Exit-flow final SepConv (conv3/conv4): dw+pw, no activation.

    BN stored separately as bn3/bn4 on the parent model (timm layout).
    Attribute names inside: conv1 (dw), pointwise (pw).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.conv1(x))
        return cast(Tensor, self.pointwise(x))


# ---------------------------------------------------------------------------
# Xception output dataclass
# ---------------------------------------------------------------------------


@dataclass
class XceptionOutput:
    """Xception classification output."""

    logits: Tensor
    loss: Tensor | None = None


# ---------------------------------------------------------------------------
# Shared forward helper
# ---------------------------------------------------------------------------


def _xception_forward_features(model: nn.Module, x: Tensor) -> Tensor:
    """Common feature-extraction path for Xception backbone and classifier."""
    # Stem
    x = F.relu(cast(Tensor, model.bn1(cast(Tensor, model.conv1(x)))))
    x = F.relu(cast(Tensor, model.bn2(cast(Tensor, model.conv2(x)))))

    # Entry flow
    x = cast(Tensor, model.block1(x))
    x = cast(Tensor, model.block2(x))
    x = cast(Tensor, model.block3(x))

    # Middle flow
    x = cast(Tensor, model.block4(x))
    x = cast(Tensor, model.block5(x))
    x = cast(Tensor, model.block6(x))
    x = cast(Tensor, model.block7(x))
    x = cast(Tensor, model.block8(x))
    x = cast(Tensor, model.block9(x))
    x = cast(Tensor, model.block10(x))
    x = cast(Tensor, model.block11(x))

    # Exit flow
    x = cast(Tensor, model.block12(x))
    x = F.relu(cast(Tensor, model.bn3(cast(Tensor, model.conv3(x)))))
    x = F.relu(cast(Tensor, model.bn4(cast(Tensor, model.conv4(x)))))

    return cast(Tensor, model.avgpool(x))


def _build_xception_body(ic: int) -> dict[str, nn.Module]:
    """Return a dict of named sub-modules matching timm legacy_xception layout."""
    return {
        # Stem (timm: conv1.weight, bn1.*, conv2.weight, bn2.*)
        "conv1": nn.Conv2d(ic, 32, 3, stride=2, padding=1, bias=False),
        "bn1": nn.BatchNorm2d(32),
        "conv2": nn.Conv2d(32, 64, 3, padding=1, bias=False),
        "bn2": nn.BatchNorm2d(64),
        # Entry flow
        "block1": _make_entry_block(64, 128, activate_first=False),
        "block2": _make_entry_block(128, 256, activate_first=True),
        "block3": _make_entry_block(256, 728, activate_first=True),
        # Middle flow (8 blocks: block4–block11)
        "block4": _make_middle_block(728),
        "block5": _make_middle_block(728),
        "block6": _make_middle_block(728),
        "block7": _make_middle_block(728),
        "block8": _make_middle_block(728),
        "block9": _make_middle_block(728),
        "block10": _make_middle_block(728),
        "block11": _make_middle_block(728),
        # Exit flow block
        "block12": _make_exit_block(),
        # Exit SepConvs (named conv3/conv4 with bn3/bn4 at model level)
        "conv3": _ExitSepConv(1024, 1536),
        "bn3": nn.BatchNorm2d(1536),
        "conv4": _ExitSepConv(1536, 2048),
        "bn4": nn.BatchNorm2d(2048),
        # Pooling
        "avgpool": nn.AdaptiveAvgPool2d((1, 1)),
    }


# ---------------------------------------------------------------------------
# Xception backbone (task="base")
# ---------------------------------------------------------------------------


class Xception(PretrainedModel, BackboneMixin):
    """Xception feature extractor — outputs (B, 2048, 1, 1) for 299×299 inputs."""

    config_class: ClassVar[type[XceptionConfig]] = XceptionConfig
    base_model_prefix: ClassVar[str] = "xception"

    def __init__(self, config: XceptionConfig) -> None:
        super().__init__(config)
        for name, module in _build_xception_body(config.in_channels).items():
            setattr(self, name, module)

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=128, reduction=4),
            FeatureInfo(stage=2, num_channels=256, reduction=8),
            FeatureInfo(stage=3, num_channels=728, reduction=16),
            FeatureInfo(stage=4, num_channels=2048, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        return _xception_forward_features(self, x)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# Xception for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class XceptionForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """Xception with Dropout + FC classification head."""

    config_class: ClassVar[type[XceptionConfig]] = XceptionConfig
    base_model_prefix: ClassVar[str] = "xception"

    def __init__(self, config: XceptionConfig) -> None:
        super().__init__(config)
        for name, module in _build_xception_body(config.in_channels).items():
            setattr(self, name, module)
        self._build_classifier(2048, config.num_classes, dropout=config.dropout)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> XceptionOutput:
        x = _xception_forward_features(self, x)
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return XceptionOutput(logits=logits, loss=loss)
