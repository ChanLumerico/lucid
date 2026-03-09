from dataclasses import dataclass

import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

from lucid.models.base import PreTrainedModelMixin
from lucid.models.vision.resnet import ResNet, resnet_50, resnet_101

__all__ = ["FCN", "FCNConfig", "fcn_resnet_50", "fcn_resnet_101"]


@dataclass
class FCNConfig:
    num_classes: int
    backbone: str = "resnet_50"
    in_channels: int = 3
    aux_loss: bool = True

    out_in_channels: int = 2048
    aux_in_channels: int = 1024

    classifier_hidden_channels: int = 512
    aux_hidden_channels: int = 256

    dropout: float = 0.1


class _FCNHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _set_conv2d_attr(
    conv: nn.Conv2d,
    *,
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] | None = None,
    dilation: int | tuple[int, int] | None = None,
) -> None:
    if stride is not None:
        conv.stride = _to_2tuple(stride)
    if padding is not None:
        conv.padding = _to_2tuple(padding)
    if dilation is not None:
        conv.dilation = _to_2tuple(dilation)


def _patch_backbone_for_fcn(backbone: ResNet) -> None:
    def _patch_stage(stage: nn.Sequential, dilation: int) -> None:
        for idx, block in enumerate(stage):
            if not hasattr(block, "conv2") or not hasattr(block.conv2, "conv"):
                raise TypeError(
                    "FCN currently expects Bottleneck-style ResNet backbones."
                )

            conv2 = block.conv2.conv
            _set_conv2d_attr(conv2, padding=dilation, dilation=dilation)

            if idx == 0:
                _set_conv2d_attr(conv2, stride=1)
                if getattr(block, "downsample", None) is not None:
                    downsample_conv = block.downsample[0]
                    _set_conv2d_attr(downsample_conv, stride=1)

    _patch_stage(backbone.layer3, dilation=2)
    _patch_stage(backbone.layer4, dilation=4)


class _FCNResNetBackbone(nn.Module):
    def __init__(self, body: ResNet) -> None:
        super().__init__()
        self.conv1 = body.stem[0]
        self.bn1 = body.stem[1]
        self.relu = body.stem[2]
        self.maxpool = body.maxpool

        self.layer1 = body.layer1
        self.layer2 = body.layer2
        self.layer3 = body.layer3
        self.layer4 = body.layer4

    @classmethod
    def from_config(cls, config: FCNConfig) -> _FCNResNetBackbone:
        builders = {"resnet_50": resnet_50, "resnet_101": resnet_101}
        if config.backbone not in builders:
            raise ValueError(f"Unsupported backbone: '{config.backbone}'")

        body = builders[config.backbone](
            num_classes=1000, in_channels=config.in_channels
        )
        _patch_backbone_for_fcn(body)
        return cls(body)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        aux = self.layer3(x)
        out = self.layer4(aux)

        return {"out": out, "aux": aux}


class _FCNSegmentationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        aux_classifier: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Tensor | dict[str, Tensor]:
        input_size = x.shape[-2:]
        features = self.backbone(x)

        out = self.classifier(features["out"])
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        if self.aux_classifier is None:
            return out

        aux = self.aux_classifier(features["aux"])
        aux = F.interpolate(aux, size=input_size, mode="bilinear", align_corners=False)

        if return_aux:
            return {"out": out, "aux": aux}
        return out


class FCN(_FCNSegmentationModel, PreTrainedModelMixin):
    def __init__(self, config: FCNConfig) -> None:
        backbone = _FCNResNetBackbone.from_config(config)
        classifier = _FCNHead(
            in_channels=config.out_in_channels,
            hidden_channels=config.classifier_hidden_channels,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

        aux_classifier = None
        if config.aux_loss:
            aux_classifier = _FCNHead(
                in_channels=config.aux_in_channels,
                hidden_channels=config.aux_hidden_channels,
                num_classes=config.num_classes,
                dropout=config.dropout,
            )

        super().__init__(backbone, classifier, aux_classifier)
        self.config = config


@register_model
def fcn_resnet_50(
    num_classes: int = 21, in_channels: int = 3, aux_loss: bool = True, **kwargs
) -> FCN:
    return FCN(
        FCNConfig(
            num_classes=num_classes,
            backbone="resnet_50",
            in_channels=in_channels,
            aux_loss=aux_loss,
            out_in_channels=2048,
            aux_in_channels=1024,
            **kwargs,
        )
    )


@register_model
def fcn_resnet_101(
    num_classes: int = 21, in_channels: int = 3, aux_loss: bool = True, **kwargs
) -> FCN:
    return FCN(
        FCNConfig(
            num_classes=num_classes,
            backbone="resnet_101",
            in_channels=in_channels,
            aux_loss=aux_loss,
            out_in_channels=2048,
            aux_in_channels=1024,
            **kwargs,
        )
    )
