import lucid.nn as nn

import lucid
from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["Inception"]


class Inception(nn.Module):  # Beta
    def __init__(self):
        super().__init__()


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
        self.dropout = nn.Dropout2d(p=0.7)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = self.conv(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)

        return x


class Inception_V1(Inception): ...
