import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["Xception", "xception"]


class _Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reps: int,
        stride: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ) -> None:
        super().__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        channels = in_channels
        if grow_first:
            rep.append(nn.ReLU())
            rep.append(
                nn.DepthSeparableConv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_channels))
            channels = out_channels

        for i in range(reps - 1):
            rep.append(nn.ReLU())
            rep.append(
                nn.DepthSeparableConv2d(
                    channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(channels))

        if not grow_first:
            rep.append(nn.ReLU())
            rep.append(
                nn.DepthSeparableConv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_channels))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU()

        if stride != 1:
            rep.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, x: Tensor) -> Tensor:
        out = self.rep(x)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        out += skip
        return out


class Xception(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.ConvBNReLU2d(
            3, 32, kernel_size=3, stride=2, padding=0, conv_bias=False
        )
        self.conv2 = nn.ConvBNReLU2d(32, 64, kernel_size=3, conv_bias=False)

        self.block1 = _Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = _Block(128, 256, reps=2, stride=2)
        self.block3 = _Block(256, 728, reps=2, stride=2)

        self.mid_blocks = nn.Sequential(*[_Block(728, 728, reps=3) for _ in range(8)])
        self.end_block = _Block(728, 1024, reps=2, stride=2, grow_first=False)

        self.conv3 = nn.DepthSeparableConv2d(1024, 1536, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = nn.DepthSeparableConv2d(1536, 2048, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.block3(self.block2(self.block1(x)))
        x = self.mid_blocks(x)
        x = self.end_block(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


@register_model
def xception(num_classes: int = 1000, **kwargs) -> Xception:
    return Xception(num_classes, **kwargs)
