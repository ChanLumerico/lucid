from dataclasses import dataclass

import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

__all__ = ["ZFNet", "ZFNetConfig", "zfnet"]


@dataclass
class ZFNetConfig:
    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5
    classifier_hidden_features: tuple[int, int] = (4096, 4096)

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("ZFNet num_classes must be greater than 0.")

        if self.in_channels <= 0:
            raise ValueError("ZFNet in_channels must be greater than 0.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("ZFNet dropout must be in the range [0.0, 1.0).")

        if len(self.classifier_hidden_features) != 2:
            raise ValueError(
                "ZFNet classifier_hidden_features must contain exactly 2 values."
            )

        if any(hidden_dim <= 0 for hidden_dim in self.classifier_hidden_features):
            raise ValueError(
                "ZFNet classifier_hidden_features values must be greater than 0."
            )


class ZFNet(nn.Module):
    def __init__(self, config: ZFNetConfig) -> None:
        super().__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(config.in_channels, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        hidden_1, hidden_2 = config.classifier_hidden_features
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(256 * 6 * 6, hidden_1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, config.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


@register_model
def zfnet(num_classes: int = 1000, **kwargs) -> ZFNet:
    config = ZFNetConfig(num_classes=num_classes, **kwargs)
    return ZFNet(config)
