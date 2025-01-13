import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["MobileNet", "mobilenet"]


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


@register_model
def mobilenet(
    width_multiplier: float = 1.0, num_classes: int = 1000, **kwargs
) -> MobileNet:
    return MobileNet(width_multiplier, num_classes, **kwargs)
