import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["MobileNet", "MobileNet_V2", "mobilenet", "mobilenet_v2"]


class _Depthwise(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class MobileNet(nn.Module):
    def __init__(self, width_multiplier: float, num_classes: int = 1000) -> None:
        super().__init__()
        alpha = width_multiplier

        self.conv1 = nn.ConvBNReLU2d(
            3, int(32 * alpha), kernel_size=3, stride=2, padding=1
        )
        self.conv2 = _Depthwise(int(32 * alpha), int(64 * alpha))

        self.conv3 = nn.Sequential(
            _Depthwise(int(64 * alpha), int(128 * alpha), stride=2),
            _Depthwise(int(128 * alpha), int(128 * alpha), stride=1),
        )
        self.conv4 = nn.Sequential(
            _Depthwise(int(128 * alpha), int(256 * alpha), stride=2),
            _Depthwise(int(256 * alpha), int(256 * alpha), stride=1),
        )

        self.conv5 = nn.Sequential(
            _Depthwise(int(256 * alpha), int(512 * alpha), stride=2),
            *[
                _Depthwise(int(512 * alpha), int(512 * alpha), stride=1)
                for _ in range(5)
            ]
        )
        self.conv6 = _Depthwise(int(512 * alpha), int(1024 * alpha), stride=2)
        self.conv7 = _Depthwise(int(1024 * alpha), int(1024 * alpha), stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class _InvertedBottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, t: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(),
        )
        depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels * t,
                in_channels * t,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * t,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(),
        )
        pointwise = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        residual_list = []
        if t > 1:
            residual_list += [expand]
        residual_list += [depthwise, pointwise]

        self.residual = nn.Sequential(*residual_list)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1 and self.in_channels == self.out_channels:
            out = self.residual(x) + x
        else:
            out = self.residual(x)
        return out


class MobileNet_V2(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        self.conv_first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
        )

        self.bottlenecks = nn.Sequential(
            self._make_stage(32, 16, t=1, n=1),
            self._make_stage(16, 24, t=6, n=2, stride=2),
            self._make_stage(24, 32, t=6, n=3, stride=2),
            self._make_stage(32, 64, t=6, n=4, stride=2),
            self._make_stage(64, 96, t=6, n=3),
            self._make_stage(96, 160, t=6, n=3, stride=2),
            self._make_stage(160, 320, t=6, n=1),
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, num_classes))

    def _make_stage(
        self, in_channels: int, out_channels: int, t: int, n: int, stride: int = 1
    ) -> nn.Sequential:
        layers = [_InvertedBottleneck(in_channels, out_channels, t, stride=stride)]
        in_channels = out_channels
        for _ in range(n - 1):
            layers.append(_InvertedBottleneck(in_channels, out_channels, t, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_first(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


@register_model
def mobilenet(
    width_multiplier: float = 1.0, num_classes: int = 1000, **kwargs
) -> MobileNet:
    return MobileNet(width_multiplier, num_classes, **kwargs)


@register_model
def mobilenet_v2(num_classes: int = 1000, **kwargs) -> MobileNet_V2:
    return MobileNet_V2(num_classes, **kwargs)
