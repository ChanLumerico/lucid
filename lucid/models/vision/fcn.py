from dataclasses import dataclass
from typing import Any

import lucid
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


class _FCNHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _FCNResNetBackbone(nn.Module):
    def __init__(self, config: FCNConfig) -> None:
        super().__init__()
        builders = {"resnet_50": resnet_50, "resnet_101": resnet_101}
        if config.backbone not in builders:
            raise ValueError(f"Unsupported backbone: '{config.backbone}'")

        builder = builders[config.backbone]
        self.body: ResNet = builder(num_classes=1000, in_channels=config.in_channels)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.body.stem(x)
        x = self.body.maxpool(x)

        x = self.body.layer1(x)
        x = self.body.layer2(x)
        aux = self.body.layer3(x)
        out = self.body.layer4(aux)

        return {"out": out, "aux": aux}


class FCN(PreTrainedModelMixin, nn.Module):
    def __init__(self, config: FCNConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = _FCNResNetBackbone(config)
        self.classifier = _FCNHead(
            in_channels=config.out_in_channels,
            hidden_channels=config.classifier_hidden_channels,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

        NotImplemented
        ...
