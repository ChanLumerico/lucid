from typing import Optional, Tuple, override

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["Inception", "inception_v1"]


class Inception(nn.Module):
    def __init__(self, num_classes: int, use_aux: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux

    @override
    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        return super().forward(x)


class _InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_1x1: int,
        reduce_3x3: int,
        out_channels_3x3: int,
        reduce_5x5: int,
        out_channels_5x5: int,
        out_channels_pool: int,
    ) -> None:
        super().__init__()

        self.branch1 = nn.ConvBNReLU2d(in_channels, out_channels_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, reduce_3x3, kernel_size=1),
            nn.ConvBNReLU2d(reduce_3x3, out_channels_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, reduce_5x5, kernel_size=1),
            nn.ConvBNReLU2d(reduce_5x5, out_channels_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ConvBNReLU2d(in_channels, out_channels_pool, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )


class _InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.ConvBNReLU2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = self.conv(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)

        return x


class Inception_V1(Inception):
    def __init__(self, num_classes: int = 1000, use_aux: bool = True) -> None:
        super().__init__(num_classes, use_aux)
        in_channels = 3

        self.conv1 = nn.ConvBNReLU2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.ConvBNReLU2d(64, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 192, kernel_size=3, stride=1, padding=1),
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incep_3 = nn.Sequential(
            _InceptionModule(192, 64, 96, 128, 16, 32, 32),
            _InceptionModule(256, 128, 128, 192, 32, 96, 64),
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incep_4a = _InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.incep_4bcd = nn.Sequential(
            _InceptionModule(512, 160, 112, 224, 24, 64, 64),
            _InceptionModule(512, 128, 128, 256, 24, 64, 64),
            _InceptionModule(512, 112, 144, 288, 32, 64, 64),
        )
        self.incep_4e = _InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.incep_5 = nn.Sequential(
            _InceptionModule(832, 256, 160, 320, 32, 128, 128),
            _InceptionModule(832, 384, 192, 384, 48, 128, 128),
        )

        if use_aux:
            self.aux1 = _InceptionAux(512, num_classes)
            self.aux2 = _InceptionAux(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))

        x = self.maxpool3(self.incep_3(x))
        x = self.incep_4a(x)
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.incep_4bcd(x)
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.maxpool4(self.incep_4e(x))
        x = self.avgpool(self.incep_5(x))

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux2, aux1


class _InceptionModule_V2A(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=3),
            nn.ConvBNReLU2d(64, 96, kernel_size=3, padding=1),
            nn.ConvBNReLU2d(96, 96, kernel_size=3, padding=1),
        )

        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 48, kernel_size=1),
            nn.ConvBNReLU2d(48, 64, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
        )

        self.branch4 = nn.ConvBNReLU2d(in_channels, 64, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )


class _InceptionModule_V2B(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, out_channels, kernel_size=1),
            nn.ConvBNReLU2d(
                out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)
            ),
            nn.ConvBNReLU2d(
                out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)
            ),
            nn.ConvBNReLU2d(
                out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)
            ),
            nn.ConvBNReLU2d(out_channels, 192, kernel_size=(7, 1), padding=(3, 0)),
        )

        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, out_channels, kernel_size=1),
            nn.ConvBNReLU2d(
                out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)
            ),
            nn.ConvBNReLU2d(out_channels, 192, kernel_size=(7, 1), padding=(3, 0)),
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
        )

        self.branch4 = nn.ConvBNReLU2d(in_channels, 192, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )


class _InceptionModule_V2C(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1_stem = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 448, kernel_size=1),
            nn.ConvBNReLU2d(448, 384, kernel_size=3, padding=1),
        )
        self.branch1_left = nn.ConvBNReLU2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1)
        )
        self.branch1_right = nn.ConvBNReLU2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0)
        )

        self.branch2_stem = nn.ConvBNReLU2d(in_channels, 384, kernel_size=1)
        self.branch2_left = nn.ConvBNReLU2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1)
        )
        self.branch2_right = nn.ConvBNReLU2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
        )

        self.branch4 = nn.ConvBNReLU2d(in_channels, 320, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1_stem = self.branch1_stem(x)
        branch2_stem = self.branch2_stem(x)

        # TODO: Begin from here


@register_model
def inception_v1(
    num_classes: int = 1000,
    use_aux: bool = True,
    **kwargs,
) -> Inception:
    return Inception_V1(num_classes, use_aux, **kwargs)
