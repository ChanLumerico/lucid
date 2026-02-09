from typing import Literal

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

from .inception import _InceptionStem_V4, _InceptionReduce_V4A


__all__ = ["InceptionResNet", "inception_resnet_v1", "inception_resnet_v2"]


class InceptionResNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.stem: nn.Module
        self.conv: nn.Sequential
        self.fc: nn.Sequential


class _InceptionResStem(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=3, stride=2),
            nn.ConvBNReLU2d(32, 32, kernel_size=3),
            nn.ConvBNReLU2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ConvBNReLU2d(64, 80, kernel_size=1, padding=0),
            nn.ConvBNReLU2d(80, 192, kernel_size=3),
            nn.ConvBNReLU2d(192, 256, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _InceptionResModule_A(nn.Module):
    def __init__(self, in_channels: int, version: Literal["v1", "v2"]) -> None:
        super().__init__()

        if version == "v1":
            cfg = [32, 32, 256]
        elif version == "v2":
            cfg = [48, 64, 384]
        else:
            raise ValueError("Invalid version.")

        self.branch1 = nn.ConvBNReLU2d(in_channels, 32, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=1),
            nn.ConvBNReLU2d(32, 32, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=1),
            nn.ConvBNReLU2d(32, cfg[0], kernel_size=3, padding=1),
            nn.ConvBNReLU2d(cfg[0], cfg[1], kernel_size=3, padding=1),
        )

        self.conv_linear = nn.Conv2d(32 + 32 + cfg[1], cfg[2], kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = lucid.concatenate([branch1, branch2, branch3], axis=1)

        out = self.conv_linear(out)
        out = out + residual
        out = self.relu(out)

        return out


class _InceptionResModule_B(nn.Module):
    def __init__(self, in_channels: int, version: Literal["v1", "v2"]) -> None:
        super().__init__()

        if version == "v1":
            cfg = [128, 128, 128, 896]
        elif version == "v2":
            cfg = [192, 160, 192, 1152]
        else:
            raise ValueError("Invalid version.")

        self.branch1 = nn.ConvBNReLU2d(in_channels, cfg[0], kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 128, kernel_size=1),
            nn.ConvBNReLU2d(128, cfg[1], kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(cfg[1], cfg[2], kernel_size=(7, 1), padding=(3, 0)),
        )

        self.conv_linear = nn.Conv2d(cfg[0] + cfg[2], cfg[3], kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = lucid.concatenate([branch1, branch2], axis=1)

        out = self.conv_linear(out)
        out = out + residual
        out = self.relu(out)

        return out


class _InceptionResModule_C(nn.Module):
    def __init__(self, in_channels: int, version: Literal["v1", "v2"]) -> None:
        super().__init__()

        if version == "v1":
            cfg = [192, 192, 1792]
        elif version == "v2":
            cfg = [224, 256, 2048]
        else:
            raise ValueError("Invalid version.")

        self.branch1 = nn.ConvBNReLU2d(in_channels, 192, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
            nn.ConvBNReLU2d(192, cfg[0], kernel_size=(1, 3), padding=(0, 1)),
            nn.ConvBNReLU2d(cfg[0], cfg[1], kernel_size=(3, 1), padding=(1, 0)),
        )

        self.conv_linear = nn.Conv2d(192 + cfg[1], in_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        out = lucid.concatenate([branch1, branch2], axis=1)

        out = self.conv_linear(out)
        out = out + residual
        out = self.relu(out)

        return out


class _InceptionResReduce(nn.Module):
    def __init__(self, in_channels: int, version: Literal["v1", "v2"]) -> None:
        super().__init__()

        if version == "v1":
            cfg = [256, 256, 256]
        elif version == "v2":
            cfg = [288, 288, 320]
        else:
            raise ValueError("Invalid version.")

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
            nn.ConvBNReLU2d(256, 384, kernel_size=3, stride=2),
        )

        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
            nn.ConvBNReLU2d(256, cfg[0], kernel_size=3, stride=2),
        )

        self.branch4 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
            nn.ConvBNReLU2d(256, cfg[1], kernel_size=3, padding=1),
            nn.ConvBNReLU2d(cfg[1], cfg[2], kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )


class InceptionResNet_V1(InceptionResNet):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__(num_classes)
        in_channels = 3

        self.stem = _InceptionResStem(in_channels)

        modules = []
        for _ in range(5):
            modules.append(_InceptionResModule_A(256, version="v1"))
        modules.append(_InceptionReduce_V4A(256, k=192, l=192, m=256, n=384))

        for _ in range(10):
            modules.append(_InceptionResModule_B(896, version="v1"))
        modules.append(_InceptionResReduce(896, version="v1"))

        for _ in range(5):
            modules.append(_InceptionResModule_C(1792, version="v1"))

        self.conv = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(1792, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class InceptionResNet_V2(InceptionResNet):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__(num_classes)
        in_channels = 3

        self.stem = _InceptionStem_V4(in_channels)

        modules = []
        for _ in range(5):
            modules.append(_InceptionResModule_A(384, version="v2"))
        modules.append(_InceptionReduce_V4A(384, k=256, l=256, m=384, n=384))

        for _ in range(10):
            modules.append(_InceptionResModule_B(1152, version="v2"))
        modules.append(_InceptionResReduce(1152, version="v2"))

        for _ in range(5):
            modules.append(_InceptionResModule_C(2144, version="v2"))

        self.conv = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(2144, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


@register_model
def inception_resnet_v1(num_classes: int = 1000, **kwargs) -> InceptionResNet:
    return InceptionResNet_V1(num_classes, **kwargs)


@register_model
def inception_resnet_v2(num_classes: int = 1000, **kwargs) -> InceptionResNet:
    return InceptionResNet_V2(num_classes, **kwargs)
