from typing import Optional, Tuple, override

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["Inception", "inception_v1", "inception_v3", "inception_v4"]


class Inception(nn.Module):
    def __init__(self, num_classes: int, use_aux: bool | None = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_aux = use_aux

    @override
    def forward(self, x: Tensor) -> Tuple[Tensor | None, ...]:
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
    def __init__(
        self, in_channels: int, num_classes: int, pool_size: tuple[int, int]
    ) -> None:
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(pool_size)
        self.conv = nn.ConvBNReLU2d(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(128 * pool_size[0] * pool_size[1], 1024)
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
            self.aux1 = _InceptionAux(512, num_classes, pool_size=(4, 4))
            self.aux2 = _InceptionAux(528, num_classes, pool_size=(4, 4))
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
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
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

        branch1 = lucid.concatenate(
            [self.branch1_left(branch1_stem), self.branch1_right(branch1_stem)],
            axis=1,
        )
        branch2 = lucid.concatenate(
            [self.branch2_left(branch2_stem), self.branch2_right(branch2_stem)],
            axis=1,
        )
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return lucid.concatenate([branch1, branch2, branch3, branch4], axis=1)


class _InceptionReduce_V2A(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 96, kernel_size=3, padding=1),
            nn.ConvBNReLU2d(96, 96, kernel_size=3, stride=2),
        )

        self.branch2 = nn.ConvBNReLU2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x)], axis=1
        )


class _InceptionReduce_V2B(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
            nn.ConvBNReLU2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            nn.ConvBNReLU2d(192, 192, kernel_size=3, stride=2),
        )

        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
            nn.ConvBNReLU2d(192, 320, kernel_size=3, stride=2),
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x)], axis=1
        )


class Inception_V3(Inception):
    def __init__(
        self,
        num_classes: int = 1000,
        use_aux: bool = True,
        dropout_prob: float = 0.5,
    ) -> None:
        super().__init__(num_classes, use_aux)
        in_channels = 3

        self.conv1 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=3, stride=2),
            nn.ConvBNReLU2d(32, 32, kernel_size=3),
            nn.ConvBNReLU2d(32, 64, kernel_size=3, padding=1),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            nn.ConvBNReLU2d(64, 80, kernel_size=3),
            nn.ConvBNReLU2d(80, 192, kernel_size=3, stride=2),
            nn.ConvBNReLU2d(192, 288, kernel_size=3, padding=1),
        )

        self.incep_3 = nn.Sequential(
            _InceptionModule_V2A(288),
            _InceptionModule_V2A(288),
            _InceptionModule_V2A(288),
        )
        self.incep_red1 = _InceptionReduce_V2A(288)

        self.incep_4 = nn.Sequential(
            _InceptionModule_V2B(768, 128),
            _InceptionModule_V2B(768, 160),
            _InceptionModule_V2B(768, 160),
            _InceptionModule_V2B(768, 160),
            _InceptionModule_V2B(768, 192),
        )
        self.incep_red2 = _InceptionReduce_V2B(768)

        if use_aux:
            self.aux = _InceptionAux(768, num_classes, pool_size=(5, 5))
        else:
            self.aux = None

        self.incep_5 = nn.Sequential(
            _InceptionModule_V2C(1280), _InceptionModule_V2C(2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)

        x = self.incep_3(x)
        x = self.incep_red1(x)
        x = self.incep_4(x)

        if self.aux is not None and self.training:
            aux = self.aux(x)
        else:
            aux = None

        x = self.incep_red2(x)
        x = self.avgpool(self.incep_5(x))

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux if self.aux is not None else x


class _InceptionStem_V4(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 32, kernel_size=3, stride=2),
            nn.ConvBNReLU2d(32, 32, kernel_size=3),
            nn.ConvBNReLU2d(32, 64, kernel_size=3, padding=1),
        )

        self.branch1_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch1_conv = nn.ConvBNReLU2d(64, 96, kernel_size=3, stride=2)

        self.branch2_left = nn.Sequential(
            nn.ConvBNReLU2d(160, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 96, kernel_size=3),
        )
        self.branch2_right = nn.Sequential(
            nn.ConvBNReLU2d(160, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            nn.ConvBNReLU2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(64, 96, kernel_size=3),
        )

        self.branch3_conv = nn.ConvBNReLU2d(192, 192, kernel_size=3, stride=2)
        self.branch3_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        x = lucid.concatenate([self.branch1_pool(x), self.branch1_conv(x)], axis=1)
        x = lucid.concatenate([self.branch2_left(x), self.branch2_left(x)], axis=1)
        x = lucid.concatenate([self.branch3_pool(x), self.branch3_conv(x)], axis=1)

        return x


class _InceptionModule_V4A(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1),
            nn.ConvBNReLU2d(in_channels, 96, kernel_size=1),
        )
        self.branch2 = nn.ConvBNReLU2d(in_channels, 96, kernel_size=1)

        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 96, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 96, kernel_size=3, padding=1),
            nn.ConvBNReLU2d(96, 96, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )


class _InceptionModule_V4B(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1),
            nn.ConvBNReLU2d(in_channels, 128, kernel_size=1),
        )

        self.branch2 = nn.ConvBNReLU2d(in_channels, 384, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
            nn.ConvBNReLU2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
        )

        self.branch4 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 64, kernel_size=1),
            nn.ConvBNReLU2d(64, 192, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            nn.ConvBNReLU2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            axis=1,
        )


class _InceptionModule_V4C(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1),
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
        )
        self.branch2 = nn.ConvBNReLU2d(in_channels, 256, kernel_size=1)

        self.branch3 = nn.ConvBNReLU2d(in_channels, 384, kernel_size=1)
        self.branch3_left = nn.ConvBNReLU2d(
            384, 256, kernel_size=(1, 3), padding=(0, 1)
        )
        self.branch3_right = nn.ConvBNReLU2d(
            384, 256, kernel_size=(3, 1), padding=(1, 0)
        )

        self.branch4 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 384, kernel_size=1),
            nn.ConvBNReLU2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            nn.ConvBNReLU2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch4_left = nn.ConvBNReLU2d(
            512, 256, kernel_size=(3, 1), padding=(1, 0)
        )
        self.branch4_right = nn.ConvBNReLU2d(
            512, 256, kernel_size=(1, 3), padding=(0, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x3 = self.branch3(x)
        x3l = self.branch3_left(x3)
        x3r = self.branch3_right(x3)

        x4 = self.branch4(x)
        x4l = self.branch4_left(x4)
        x4r = self.branch4_right(x4)

        return lucid.concatenate([x1, x2, x3l, x3r, x4l, x4r], axis=1)


class _InceptionReduce_V4A(nn.Module):
    def __init__(self, in_channels: int, k: int, l: int, m: int, n: int) -> None:
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch2 = nn.ConvBNReLU2d(in_channels, n, kernel_size=3, stride=2)

        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, k, kernel_size=1),
            nn.ConvBNReLU2d(k, l, kernel_size=3, padding=1),
            nn.ConvBNReLU2d(l, m, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x)],
            axis=1,
        )


class _InceptionReduce_V4B(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.branch1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 192, kernel_size=1),
            nn.ConvBNReLU2d(192, 192, kernel_size=3, stride=2),
        )

        self.branch3 = nn.Sequential(
            nn.ConvBNReLU2d(in_channels, 256, kernel_size=1),
            nn.ConvBNReLU2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            nn.ConvBNReLU2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            nn.ConvBNReLU2d(320, 320, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return lucid.concatenate(
            [self.branch1(x), self.branch2(x), self.branch3(x)],
            axis=1,
        )


class Inception_V4(Inception):
    def __init__(self, num_classes: int = 1000, use_aux: bool = True):
        super().__init__(num_classes, use_aux)
        in_channels = 3

        modules = []
        modules.append(_InceptionStem_V4(in_channels))

        for _ in range(4):
            modules.append(_InceptionModule_V4A(384))
        modules.append(_InceptionReduce_V4A(384, k=192, l=224, m=256, n=384))

        for _ in range(7):
            modules.append(_InceptionModule_V4B(1024))
        modules.append(_InceptionReduce_V4B(1024))

        for _ in range(3):
            modules.append(_InceptionModule_V4C(1536))

        self.conv = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


@register_model
def inception_v1(
    num_classes: int = 1000,
    use_aux: bool = True,
    **kwargs,
) -> Inception:
    return Inception_V1(num_classes, use_aux, **kwargs)


@register_model
def inception_v3(
    num_classes: int = 1000,
    use_aux: bool = True,
    dropout_prob: float = 0.5,
    **kwargs,
) -> Inception:
    return Inception_V3(num_classes, use_aux, dropout_prob, **kwargs)


@register_model
def inception_v4(num_classes: int = 1000, **kwargs) -> Inception:
    return Inception_V4(num_classes, None, **kwargs)
